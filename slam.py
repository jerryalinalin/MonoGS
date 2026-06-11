import os
import random
import sys
import time

import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir

        # 固定随机种子（确保可复现）
        if "seed" in config:
            seed = config["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.save_dir = self.save_dir

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        Log("Peak VRAM [MB]", torch.cuda.max_memory_allocated() // 1024**2, tag="Eval")
        Log("VRAM at end [MB]", torch.cuda.memory_allocated() // 1024**2, tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # === Phase 2: 各向异性分析（放color refinement前以确保执行）===
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            scales = self.gaussians.get_scaling.detach().cpu().numpy()
            aniso_ratio = scales.max(axis=1) / (scales.min(axis=1) + 1e-6)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(aniso_ratio, bins=100, range=(0, 50))
            axes[0].set_xlabel("Anisotropy Ratio")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Anisotropy Distribution")
            axes[1].hist(aniso_ratio[aniso_ratio <= 20], bins=80, range=(0, 20))
            axes[1].set_xlabel("Anisotropy Ratio")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Zoomed (0-20)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "aniso_distribution.png"), dpi=150)
            plt.close()
            mean_ratio = float(aniso_ratio.mean())
            pct_gt_10 = float((aniso_ratio > 10).mean() * 100)
            pct_gt_50 = float((aniso_ratio > 50).mean() * 100)
            Log(f"[AnisoDist] mean_ratio={mean_ratio:.2f}, "
                f">10x={pct_gt_10:.1f}%, >50x={pct_gt_50:.1f}%, total={len(aniso_ratio)}", tag="Eval")
            with open(os.path.join(self.save_dir, "aniso_stats.txt"), "w") as f:
                f.write(f"mean_anisotropy_ratio={mean_ratio:.4f}\n")
                f.write(f"pct_ratio_gt_10={pct_gt_10:.2f}\n")
                f.write(f"pct_ratio_gt_50={pct_gt_50:.2f}\n")
                f.write(f"total_gaussians={len(aniso_ratio)}\n")
                f.write(f"min_ratio={float(aniso_ratio.min()):.4f}\n")
                f.write(f"max_ratio={float(aniso_ratio.max()):.4f}\n")
                f.write(f"median_ratio={float(np.median(aniso_ratio)):.4f}\n")



            # ===== C1：残差 vs 图像梯度散点图 =====
            import os as _c1_os
            import json as _c1_json
            import random as _c1_random
            import cv2
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from gaussian_splatting.gaussian_renderer import render

            Log("[C1] 开始生成残差 vs 梯度散点图...", tag="Eval")

            PATCH_SIZE = 32
            MAX_FRAMES = 20
            _cam_list = list(self.frontend.cameras.values())
            _cam_list = [c for c in _cam_list if c is not None and c.original_image is not None]
            if len(_cam_list) == 0:
                Log("[C1] 警告：没有可用相机帧，跳过", tag="Eval")
            else:
                _sampled = _c1_random.sample(_cam_list, min(MAX_FRAMES, len(_cam_list)))
                _res_list, _grad_list = [], []
                for _cam in _sampled:
                    try:
                        with torch.no_grad():
                            _rp = render(_cam, self.gaussians, self.pipeline_params, self.background)
                            _rendered = _rp["render"]
                            _gt = _cam.original_image.cuda()
                            _res_map = (_rendered - _gt).abs().mean(0).cpu().numpy()
                            _gt_gray = _gt.mean(0).cpu().numpy()
                            _gx = cv2.Sobel(_gt_gray, cv2.CV_64F, 1, 0, ksize=3)
                            _gy = cv2.Sobel(_gt_gray, cv2.CV_64F, 0, 1, ksize=3)
                            _grad_map = np.sqrt(_gx**2 + _gy**2)
                        _H, _W = _res_map.shape
                        for _i in range(0, _H - PATCH_SIZE + 1, PATCH_SIZE):
                            for _j in range(0, _W - PATCH_SIZE + 1, PATCH_SIZE):
                                _res_list.append(float(_res_map[_i:_i+PATCH_SIZE, _j:_j+PATCH_SIZE].mean()))
                                _grad_list.append(float(_grad_map[_i:_i+PATCH_SIZE, _j:_j+PATCH_SIZE].mean()))
                    except Exception as _e:
                        Log(f"[C1] 跳过帧（{_e}）", tag="Eval")
                        continue
                if len(_res_list) > 0:
                    _c1_json_path = _c1_os.path.join(self.save_dir, "density_vs_residual.json")
                    with open(_c1_json_path, 'w') as _f:
                        _c1_json.dump({"patch_residuals": _res_list, "patch_gradients": _grad_list,
                                       "patch_size": PATCH_SIZE, "num_frames": len(_sampled)}, _f, indent=2)
                    _fig, _ax = plt.subplots(figsize=(8, 6))
                    _ax.scatter(_grad_list, _res_list, alpha=0.3, s=8, color='steelblue')
                    _ax.set_xlabel("Image Gradient (Sobel)", fontsize=12)
                    _ax.set_ylabel("Render Residual (L1)", fontsize=12)
                    _ax.set_title(f"Gradient vs Residual (patch={PATCH_SIZE}px, frames={len(_sampled)})", fontsize=12)
                    _corr = np.corrcoef(_grad_list, _res_list)[0, 1]
                    _ax.text(0.05, 0.95, f"r = {_corr:.3f}", transform=_ax.transAxes, fontsize=11,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    _png = _c1_os.path.join(self.save_dir, "density_vs_residual.png")
                    plt.tight_layout(); plt.savefig(_png, dpi=150); plt.close()
                    Log(f"[C1] 完成：{_png}（{len(_res_list)} patches, r={_corr:.3f}）", tag="Eval")
                else:
                    Log("[C1] 警告：没有有效 patch 数据", tag="Eval")
            # ===== C1 结束 =====

            # ===== C2：Densification 事件时间线 =====
            Log("[C2] 开始生成 densification 时间线...", tag="Eval")
            _drecs = []
            _c2_jsonl = _c1_os.path.join(self.save_dir, "densify_records.jsonl")
            if _c1_os.path.exists(_c2_jsonl):
                with open(_c2_jsonl) as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line:
                            _drecs.append(_c1_json.loads(_line))
            if len(_drecs) == 0:
                Log("[C2] 警告：没有 densify_records，跳过", tag="Eval")
            else:
                _c2_json_path = _c1_os.path.join(self.save_dir, "densify_timeline.json")
                with open(_c2_json_path, 'w') as _f:
                    _c1_json.dump(_drecs, _f, indent=2)
                _iters = [r["iter"] for r in _drecs]
                _cnts = [r["after"] for r in _drecs]
                _stretch = [r["high_stretch_ratio_before"] for r in _drecs]
                _fig2, _ax1 = plt.subplots(figsize=(12, 5))
                _ax1.set_xlabel("Training Iteration", fontsize=11)
                _ax1.set_ylabel("Gaussian Count", color='steelblue', fontsize=11)
                _ax1.plot(_iters, _cnts, color='steelblue', linewidth=1.5, label="Gaussian count")
                _ax1.tick_params(axis='y', labelcolor='steelblue')
                _ax2 = _ax1.twinx()
                _ax2.set_ylabel("High-stretch Ratio (>10x)", color='tomato', fontsize=11)
                _ax2.plot(_iters, _stretch, color='tomato', linewidth=1.5, linestyle='--')
                _ax2.tick_params(axis='y', labelcolor='tomato')
                _ax2.set_ylim(0, 1)
                _fig2.suptitle("Densification Timeline", fontsize=12)
                _fig2.tight_layout()
                _c2_png = _c1_os.path.join(self.save_dir, "densify_timeline.png")
                plt.savefig(_c2_png, dpi=150); plt.close()
                Log(f"[C2] 完成：{_c2_png}（{len(_drecs)} 次 densify 事件）", tag="Eval")
            # ===== C2 结束 =====
            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        _dir_map = {"datasets_tum": "tum_freiburg3", "datasets_replica": "replica_room0"}
        _subdir = _dir_map.get(path[-3] + "_" + path[-2], path[-3] + "_" + path[-2])
        save_dir = os.path.join(
            config["Results"]["save_dir"], _subdir, current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
