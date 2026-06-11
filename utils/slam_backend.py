import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


# === Phase 4 B3: 各向同性正则化（阈值截断版）===
def isotropic_loss(gaussians, lambda_iso=0.01, threshold=None):
    """各向同性正则化。threshold=None 为原始版（惩罚所有），>0 为 B3 版（只惩罚超过阈值的）。"""
    if lambda_iso <= 0:
        return torch.tensor(0.0).cuda()
    scales = gaussians.get_scaling                        # [N, 3]
    s_max = scales.max(dim=1).values                      # 最长轴
    s_min = scales.min(dim=1).values                      # 最短轴
    ratio = s_max / (s_min + 1e-6)
    if threshold is not None and threshold > 0:
        # B3: 只惩罚超过阈值的部分，relu(ratio - threshold)
        loss = torch.clamp(ratio - threshold, min=0.0).mean()
    else:
        # 原始: 惩罚所有各向异性 (ratio - 1)
        loss = (ratio - 1.0).mean()
    return lambda_iso * loss


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.densify_records = []
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            rendered_acm = []
            gt_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                rendered_acm.append(image)
                gt_acm.append(viewpoint.original_image.cuda())

            # === D1: 历史帧重放 ===
            if self.config["Training"].get("keyframe_replay", False):
                _pool = [i for i in self.viewpoints.keys() if i not in current_window]
                if len(_pool) > 0:
                    _ridx = _pool[random.randint(0, len(_pool) - 1)]
                    _rcam = self.viewpoints[_ridx]
                    _rpkg = render(_rcam, self.gaussians, self.pipeline_params, self.background)
                    _rloss = get_loss_mapping(self.config, _rpkg["render"], _rpkg["depth"], _rcam, _rpkg["opacity"])
                    loss_mapping += _rloss * 0.5
                    Log(f"[KeyframeReplay] frame {_ridx} loss={_rloss.item():.4f}")

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                if self.config["Training"].get("adaptive_densify", False):
                    _factor = self.config["Training"].get("densify_residual_thresh", 1.5)
                    for _idx in range(min(1, len(rendered_acm))):
                        _res = (rendered_acm[_idx] - gt_acm[_idx]).abs().mean(0)
                        _mean_r = _res.mean()
                        _mask = _res > _mean_r * _factor
                        if not hasattr(self, "_adapt_log_cnt"): self._adapt_log_cnt = 0
                        if self._adapt_log_cnt % 500 == 0:
                            Log(f"[AdaptDens] high-residual pixels: {_mask.float().mean()*100:.1f}%")
                        self._adapt_log_cnt += 1
                        self.gaussians.boost_densification_for_mask(_mask, viewpoint_stack[0], grad_threshold=self.opt_params.densify_grad_threshold)

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    # ===== C2 记录：densify 前 =====
                    before_count = self.gaussians.get_xyz.shape[0]
                    scales = self.gaussians.get_scaling
                    s_max = scales.max(dim=1).values
                    s_min = scales.min(dim=1).values
                    ratio = s_max / (s_min + 1e-6)
                    high_stretch_before = (ratio > 10.0).float().mean().item()
                    # ===== C2 记录结束 =====

                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                    # ===== C2 记录：densify 后 =====
                    after_count = self.gaussians.get_xyz.shape[0]
                    _rec = {
                        "iter": self.iteration_count,
                        "before": before_count,
                        "after": after_count,
                        "added": after_count - before_count,
                        "high_stretch_ratio_before": round(high_stretch_before, 4),
                    }
                    if not hasattr(self, 'densify_records'):
                        self.densify_records = []
                    self.densify_records.append(_rec)
                    _sd = getattr(self, 'save_dir', None)
                    if _sd:
                        import json as _j, os
                        with open(os.path.join(_sd, "densify_records.jsonl"), "a") as _fp:
                            _fp.write(_j.dumps(_rec) + "\n")
                    # ===== C2 记录结束 =====
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        # === Phase 2: 窗口残差记录（缺陷A可视化，只读不改）===
        if len(current_window) == self.config["Training"]["window_size"] and self.iteration_count % 100 == 0:
            losses = {}
            for kf_idx in current_window:
                cam = self.viewpoints.get(kf_idx)
                if cam is None:
                    continue
                with torch.no_grad():
                    render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background)
                    rendered = render_pkg["render"]
                    gt = cam.original_image.cuda()
                    loss = torch.abs(rendered - gt).mean().item()
                losses[kf_idx] = loss
            if losses:
                sorted_l = sorted(losses.items(), key=lambda x: x[1])
                Log(f"[WindowLoss] iter={self.iteration_count} "
                    f"min={sorted_l[0][1]:.6f}(f{sorted_l[0][0]}) "
                    f"max={sorted_l[-1][1]:.6f}(f{sorted_l[-1][0]}) "
                    f"window={list(current_window)}")
                # 保存到结果目录
                if hasattr(self, 'save_dir') and self.save_dir:
                    import json, os
                    record = {"iter": self.iteration_count,
                              "losses": {str(k): v for k, v in losses.items()},
                              "min_frame": sorted_l[0][0], "min_loss": sorted_l[0][1],
                              "max_frame": sorted_l[-1][0], "max_loss": sorted_l[-1][1]}
                    fpath = os.path.join(self.save_dir, "window_loss_records.jsonl")
                    with open(fpath, "a") as f:
                        f.write(json.dumps(record) + "\n")

            # === Defect D: 历史帧残差追踪 ===
            if hasattr(self, 'save_dir') and self.save_dir:
                import json as _j, os as _o, random as _r, numpy as _np
                _ws = set(current_window)
                _his = [i for i in self.viewpoints.keys() if i not in _ws]
                if len(_his) > 0 and len(current_window) > 0:
                    with torch.no_grad():
                        _wl = []
                        for _k in current_window:
                            _c = self.viewpoints.get(_k)
                            if _c is None or _c.original_image is None: continue
                            _p = render(_c, self.gaussians, self.pipeline_params, self.background)
                            _wl.append((_p["render"] - _c.original_image.cuda()).abs().mean().item())
                        _hs = _r.sample(_his, min(5, len(_his)))
                        _hl = []
                        for _k in _hs:
                            _c = self.viewpoints.get(_k)
                            if _c is None or _c.original_image is None: continue
                            _p = render(_c, self.gaussians, self.pipeline_params, self.background)
                            _hl.append((_p["render"] - _c.original_image.cuda()).abs().mean().item())
                        _rec = {"iter": self.iteration_count,
                                "window_mean": float(_np.mean(_wl)) if _wl else None,
                                "history_mean": float(_np.mean(_hl)) if _hl else None}
                        with open(_o.path.join(self.save_dir, "historical_frame_residual.jsonl"), "a") as _fp:
                            _fp.write(_j.dumps(_rec) + "\n")
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            # === Phase 4: 各向同性正则化 ===
            if self.config["Training"].get("iso_reg", False):
                lambda_iso = self.config["Training"].get("iso_reg_lambda", 0.01)
                mode = self.config["Training"].get("iso_reg_mode", "b1")
                if mode == "b2":
                    iso_loss = isotropic_loss_view_aware(self.gaussians, viewpoint_cam, lambda_iso)
                else:
                    threshold = self.config["Training"].get("iso_reg_threshold", None)
                    iso_loss = isotropic_loss(self.gaussians, lambda_iso, threshold)
                loss = loss + iso_loss
                if self.iteration_count % 500 == 0:
                    Log(f"[IsoReg] mode={mode} lambda={lambda_iso:.4f} loss={iso_loss.item():.6f}")
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        # 子进程固定随机种子（spawn 模式不继承主进程 seed）
        if "seed" in self.config:
            seed = self.config["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return


# === Phase 4 B2: 视线方向感知的各向同性正则化 ===
def isotropic_loss_view_aware(gaussians, viewpoint_cam, lambda_iso=0.01):
    """沿视线方向的拉伸有害，垂直表面的拉伸合理。alignment 加权惩罚。"""
    if lambda_iso <= 0:
        return torch.tensor(0.0).cuda()
    from gaussian_splatting.utils.general_utils import build_rotation
    xyz = gaussians.get_xyz
    scales = gaussians.get_scaling
    rots = build_rotation(gaussians.get_rotation)
    cam_center = viewpoint_cam.camera_center.cuda()
    view_dir = xyz - cam_center.unsqueeze(0)
    view_dir = view_dir / (view_dir.norm(dim=1, keepdim=True) + 1e-8)
    axis_idx = scales.argmax(dim=1)
    batch_indices = torch.arange(scales.shape[0], device=scales.device)
    longest_axis = rots[batch_indices, :, axis_idx]
    alignment = (view_dir * longest_axis).sum(dim=1).abs()
    s_max = scales.max(dim=1).values
    s_min = scales.min(dim=1).values
    ratio = s_max / (s_min + 1e-6)
    loss = (alignment * (ratio - 1.0)).mean()
    return lambda_iso * loss
