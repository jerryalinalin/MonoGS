"""
MonoGS 环境预检脚本 v2
跑基线之前先执行：python precheck.py
全部 PASS 才跑 slam.py

覆盖的历史问题：
  #1 evo==1.11.0（align_trajectory API breaking change）
  #2 matplotlib<3.6（evo 1.11.0 colorbar 兼容性）
  #3 numpy<2（PyTorch 2.x + 旧版 evo/opencv 不兼容 numpy 2.x）
  #4 setuptools<68（pkg_resources 被移除）
  #5 opencv-python<4.9 / plyfile<1.0（numpy<2 依赖）
  #6 diff-gaussian-rasterization RMurai fork API（projmatrix_raw，5返回值）
  #7 多进程残留（运行时行为，precheck 无法检测，提示用户手动清理）
"""

import sys
import os
from packaging.version import Version

errors = []
warnings = []

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        errors.append(name)

def warn(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  WARN  {name}: {e}")
        warnings.append(name)

def version_check(pkg_name, import_name, max_version=None, exact_version=None, min_version=None):
    """版本号硬检查：不只检查能 import，还检查版本范围"""
    import importlib
    mod = importlib.import_module(import_name)
    ver_str = getattr(mod, "__version__", "unknown")
    ver = Version(ver_str)

    desc = f"{pkg_name}=={ver_str}"
    if exact_version and ver != Version(exact_version):
        raise RuntimeError(f"需要 =={exact_version}，实际 {ver_str}")
    if max_version and ver >= Version(max_version):
        raise RuntimeError(f"需要 <{max_version}，实际 {ver_str}")
    if min_version and ver < Version(min_version):
        raise RuntimeError(f"需要 >={min_version}，实际 {ver_str}")
    print(f"        {desc}")


# ─────────────────────────────────────────
print("=== 1. 版本号硬检查（历史踩坑版本）===")
# ─────────────────────────────────────────

check("numpy < 2.0  (#3)",
      lambda: version_check("numpy", "numpy", max_version="2.0"))

check("evo == 1.11.0  (#1)", lambda: (
    __import__("pkg_resources").get_distribution("evo").version == "1.11.0"
    or (_ for _ in ()).throw(RuntimeError("evo 版本不是 1.11.0"))
))

check("matplotlib < 3.6  (#2)",
      lambda: version_check("matplotlib", "matplotlib", max_version="3.6"))

check("setuptools < 68  (#4)",
      lambda: version_check("setuptools", "setuptools", max_version="68"))

check("opencv-python < 4.9  (#5)", lambda: (
    version_check("cv2", "cv2", max_version="4.9.0")
))

check("plyfile < 1.0  (#5)", lambda: (
    Version(__import__("pkg_resources").get_distribution("plyfile").version) < Version("1.0")
))

check("pkg_resources 可导入  (#4)",
      lambda: __import__("pkg_resources"))


# ─────────────────────────────────────────
print("\n=== 2. 核心依赖导入 ===")
# ─────────────────────────────────────────

for pkg, imp in [
    ("torch",                      "torch"),
    ("torchvision",                "torchvision"),
    ("cv2",                        "cv2"),
    ("open3d",                     "open3d"),
    ("lpips",                      "lpips"),
    ("trimesh",                    "trimesh"),
    ("wandb",                      "wandb"),
    ("munch",                      "munch"),
    ("rich",                       "rich"),
]:
    check(f"import {pkg}", lambda i=imp: __import__(i))


# ─────────────────────────────────────────
print("\n=== 3. CUDA 可用性 ===")
# ─────────────────────────────────────────

import torch

check("cuda available", lambda: (
    (_ for _ in ()).throw(RuntimeError("CUDA not available"))
    if not torch.cuda.is_available() else None
))

check("cuda tensor ops (matmul)", lambda: (
    torch.randn(256, 256).cuda() @ torch.randn(256, 256).cuda()
))

check("cuda memory alloc / free", lambda: (
    torch.cuda.empty_cache(),
    torch.randn(1000, 1000, device="cuda"),
    torch.cuda.empty_cache()
))

# 仅打印，不算 PASS/FAIL
try:
    print(f"        torch={torch.__version__}  "
          f"cuda_runtime={torch.version.cuda}  "
          f"gpu={torch.cuda.get_device_name(0)}  "
          f"vram={torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
except Exception:
    pass


# ─────────────────────────────────────────
print("\n=== 4. evo 关键 API  (#1) ===")
# ─────────────────────────────────────────

check("evo.core.trajectory: PosePath3D / PoseTrajectory3D", lambda:
    __import__("evo.core.trajectory",
               fromlist=["PosePath3D", "PoseTrajectory3D"])
)

check("evo.core.trajectory.align_trajectory 存在", lambda: (
    getattr(
        __import__("evo.core.trajectory", fromlist=["align_trajectory"]),
        "align_trajectory"
    )
))

check("evo.core.sync 可导入", lambda:
    __import__("evo.core.sync")
)

check("evo.core.metrics: APE / PoseRelation", lambda:
    __import__("evo.core.metrics", fromlist=["APE", "PoseRelation"])
)

check("evo.tools.file_interface: read_tum_trajectory_file", lambda:
    __import__("evo.tools.file_interface",
               fromlist=["read_tum_trajectory_file"])
)


# ─────────────────────────────────────────
print("\n=== 5. evo + matplotlib 联动测试  (#2) ===")
# ─────────────────────────────────────────

def check_evo_plot():
    """实际画一次图，触发 colorbar，捕捉 matplotlib 版本不兼容"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # 无头模式，不弹窗
    from evo.tools import plot as evo_plot
    from evo.core.trajectory import PoseTrajectory3D

    n = 20
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 1, n)
    stamps = np.linspace(0, 1, n)
    traj = PoseTrajectory3D(poses_se3=poses, timestamps=stamps)

    # evo 1.11.0: prepare_axis(fig, plot_mode) 返回 Axes 对象
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = evo_plot.prepare_axis(fig, evo_plot.PlotMode.xy)
    evo_plot.traj(ax, evo_plot.PlotMode.xy, traj, "-", "blue", "test")
    plt.close("all")

check("evo + matplotlib 画图（colorbar 兼容性）", check_evo_plot)


# ─────────────────────────────────────────
print("\n=== 6. diff-gaussian-rasterization 前向传播  (#6) ===")
# ─────────────────────────────────────────

def check_rasterizer():
    """RMurai fork：需要 projmatrix_raw，返回 5 个值"""
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    H, W = 64, 64
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3).cuda(),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4).cuda(),
        projmatrix=torch.eye(4).cuda(),
        projmatrix_raw=torch.eye(4).cuda(),   # RMurai fork 额外参数 (#6)
        sh_degree=0,
        campos=torch.zeros(3).cuda(),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    N = 10
    means3D    = torch.randn(N, 3).cuda()
    means2D    = torch.zeros(N, 2).cuda().requires_grad_(True)
    opacity    = torch.ones(N, 1).cuda()
    scales     = torch.ones(N, 3).cuda() * 0.01
    rotations  = torch.zeros(N, 4).cuda()
    rotations[:, 0] = 1.0
    shs        = torch.ones(N, 1, 3).cuda()

    # RMurai fork 返回 5 个值 (#6)
    rendered_image, radii, depth, out_opacity, n_touched = rasterizer(
        means3D=means3D, means2D=means2D,
        shs=shs, colors_precomp=None,
        opacities=opacity, scales=scales,
        rotations=rotations, cov3D_precomp=None,
    )
    assert rendered_image.shape == (3, H, W), \
        f"rendered shape mismatch: {rendered_image.shape}"
    assert radii.shape == (N,), \
        f"radii shape mismatch: {radii.shape}"

check("rasterizer forward pass (RMurai fork)", check_rasterizer)

check("simple_knn 可导入", lambda: __import__("simple_knn"))


# ─────────────────────────────────────────
print("\n=== 7. 数据集路径 ===")
# ─────────────────────────────────────────

dataset_paths = [
    ("Replica room0 (RGB-D 场景A)",
     "datasets/replica/room0"),
    ("TUM fr3_office (Mono 场景B)",
     "datasets/tum/rgbd_dataset_freiburg3_long_office_household"),
]

for name, path in dataset_paths:
    check(f"{name}  →  {path}",
          lambda p=path: (_ for _ in ()).throw(
              FileNotFoundError(f"目录不存在: {p}")
          ) if not os.path.isdir(p) else None
    )


# ─────────────────────────────────────────
print("\n=== 8. 配置文件 ===")
# ─────────────────────────────────────────

configs = [
    "configs/rgbd/replica/room0.yaml",
    "configs/mono/tum/fr3_office.yaml",
]
for cfg in configs:
    check(f"config exists: {cfg}",
          lambda c=cfg: (_ for _ in ()).throw(FileNotFoundError(c))
          if not os.path.isfile(c) else None
    )


# ─────────────────────────────────────────
print("\n=== 9. GPU 残留进程检查  (#7) ===")
# ─────────────────────────────────────────

def check_gpu_processes():
    """检查是否有残留的 slam.py 进程占用 GPU"""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if lines:
        pids = [l.split(",")[0].strip() for l in lines]
        raise RuntimeError(
            f"检测到 {len(lines)} 个进程占用 GPU（PID: {', '.join(pids)}）\n"
            f"        请先执行: kill {' '.join(pids)}"
        )

warn("无残留 GPU 进程", check_gpu_processes)


# ─────────────────────────────────────────
print("\n=== 10. 包版本清单（供存档）===")
# ─────────────────────────────────────────

import importlib.metadata
snapshot_pkgs = [
    ("numpy",       "numpy"),
    ("evo",         "evo"),
    ("matplotlib",  "matplotlib"),
    ("setuptools",  "setuptools"),
    ("cv2",         "opencv-python"),
    ("plyfile",     "plyfile"),
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("open3d",      "open3d"),
    ("lpips",       "lpips"),
]
for display, pkg_name in snapshot_pkgs:
    try:
        ver = importlib.metadata.version(pkg_name)
        print(f"        {display:20s} {ver}")
    except Exception:
        print(f"        {display:20s} (version unknown)")


# ─────────────────────────────────────────
print("\n" + "=" * 50)
# ─────────────────────────────────────────

if errors:
    print(f"FAILED — {len(errors)} 个问题需要修复:")
    for e in errors:
        print(f"  ✗  {e}")
    print()
    print("快速修复命令（按顺序执行）:")
    print("  pip install evo==1.11.0 --force-reinstall")
    print("  pip install 'matplotlib<3.6' --force-reinstall")
    print("  pip install 'numpy<2' --force-reinstall")
    print("  pip install 'setuptools<68' --force-reinstall")
    print("  pip install 'opencv-python<4.9' 'plyfile<1.0' --force-reinstall")
    print("  pip install numpy<2   # 最后再锁一次，防止被拉上去")
    sys.exit(1)

if warnings:
    print(f"WARNING — {len(warnings)} 个警告（不阻止运行，但建议处理）:")
    for w in warnings:
        print(f"  ⚠  {w}")
    print()

print("ALL PASS — 环境就绪，可以跑 slam.py")
sys.exit(0)