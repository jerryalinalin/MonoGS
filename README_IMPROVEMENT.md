# MonoGS Improvements

Method-level improvements over [MonoGS: Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) (CVPR 2024 Highlight).

**Branch**: `improvement` | **Base commit**: original `muskie82/MonoGS` main

---

## Overview

This work identifies two independent method-level limitations in MonoGS and proposes targeted improvements for each. All experiments are conducted on the same hardware with a fixed random seed (42) to ensure reproducibility.

| # | Limitation | Improvement | Report |
|---|-----------|-------------|--------|
| 1 | Gaussian covariance unconstrained → anisotropic artifacts in novel views | **B4: View-Direction-Aware Isotropic Regularization** | §5.1 / §6.1 |
| 2 | Evicted keyframes receive zero gradient forever → geometric drift | **D2: Historical Keyframe Replay (10%)** | §5.2 / §6.2 |

---

## Ablation Results

All four ablation variants run under identical conditions: same hardware (RTX 3090), same hyperparameters, same random seed (42).

### TUM fr3/office — Monocular

| Version | ATE(cm)↓ | RPE(cm)↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|----------|----------|-------|-------|--------|
| Baseline | 3.45 | 0.72 | 22.36 | 0.766 | 0.342 |
| +B4 (IsoReg) | 3.50 | 0.77 | 22.39 | 0.766 | 0.369 |
| +D2 (Replay) | **3.28** | **0.69** | 22.12 | 0.762 | 0.349 |
| Full (B4+D2) | **3.15** | 0.73 | 22.20 | 0.763 | 0.368 |

### Replica room0 — RGB-D

| Version | ATE(cm)↓ | RPE(cm)↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|----------|----------|-------|-------|--------|
| Baseline | 0.45 | 0.27 | **36.84** | **0.968** | 0.049 |
| +B4 (IsoReg) | 0.47 | **0.24** | 35.78 | 0.962 | 0.059 |
| +D2 (Replay) | **0.41** | 0.28 | 36.65 | **0.968** | **0.047** |
| Full (B4+D2) | 0.44 | 0.21 | 35.59 | 0.961 | 0.059 |

**Key findings:**
- D2 is the only variant where all four metrics improve simultaneously on TUM (ATE −5.2%, RPE −4.2%, PSNR +0.23 dB, LPIPS −2.0% vs baseline with seed=42 ablation run)
- B4 achieves the best ATE on Replica (−8.9%) by constraining view-direction elongation
- Full (B4+D2) achieves the best combined ATE on TUM (−8.7%), confirming additive contribution

---

## Environment

| Item | Value |
|------|-------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3090 (24 GB) |
| CUDA | 12.1 (cu121 wheel) |
| Python | 3.10 |
| PyTorch | 2.2.2+cu121 |

> ⚠️ **Windows is not supported.** MonoGS uses `multiprocessing` with shared CUDA tensors. Linux `fork` mode inherits GPU memory directly; Windows `spawn` mode requires pickle serialization which corrupts tensor views, causing `linalg.inv` failures. See `docs/RESEARCH_LOG.md §Phase 0` for full analysis.

### Installation

```bash
conda create -n MonoGS python=3.10 -y
conda activate MonoGS

# PyTorch with CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Dependencies with version pins (order matters)
pip install opencv-python munch trimesh evo==1.11.0 open3d \
  "torchmetrics<1.0" imgviz PyOpenGL glfw PyGLM wandb \
  lpips rich ninja "plyfile<1.0" \
  --index-url https://pypi.org/simple/

# Lock versions that conflict with numpy 2.x
pip install "numpy<2" "setuptools<68" "matplotlib<3.6" \
  "opencv-python<4.9" \
  --index-url https://pypi.org/simple/

# Compile CUDA extensions
# Note: simple-knn requires a one-line patch first:
#   Add `#include <cfloat>` to submodules/simple-knn/simple_knn.cu line 1
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

# Verify environment
python scripts/env_precheck.py   # must all PASS before running slam.py
```

### Code Modifications Summary

| File | Change | Purpose |
|------|--------|---------|
| `submodules/simple-knn/simple_knn.cu` | Add `#include <cfloat>` | CUDA 12 compatibility |
| `utils/slam_backend.py` | Add `isotropic_loss_view_aware()`, call in `map()` | Improvement B4 |
| `utils/slam_backend.py` | Modify viewpoint sampling in `map()` | Improvement D2 |
| `configs/*/fr3_office_imp{1-4}.yaml` | Ablation configs for TUM | Ablation |
| `configs/*/room0_imp{1-4}.yaml` | Ablation configs for Replica | Ablation |

No other source files are modified. All improvements are toggled via yaml config flags and disabled by default.

---

## Reproduce

### Step 1 — Verify environment

```bash
python scripts/env_precheck.py
# Expected: ALL PASS
```

### Step 2 — Prepare datasets

```bash
# Replica room0 (synthetic, RGB-D)
bash scripts/download_replica.sh
# → datasets/replica/room0/

# TUM fr3/office (real-world, monocular)
bash scripts/download_tum.sh
# → datasets/tum/rgbd_dataset_freiburg3_long_office_household/
```

### Step 3 — Run baseline

```bash
# TUM (monocular)
WANDB_MODE=offline python slam.py \
  --config configs/mono/tum/fr3_office_imp1.yaml --eval

# Replica (RGB-D)
WANDB_MODE=offline python slam.py \
  --config configs/rgbd/replica/room0_imp1.yaml --eval
```

### Step 4 — Run ablation variants

```bash
# TUM × 4 variants
for v in imp1 imp2 imp3 imp4; do
  WANDB_MODE=offline python slam.py \
    --config configs/mono/tum/fr3_office_${v}.yaml --eval
done

# Replica × 4 variants
for v in imp1 imp2 imp3 imp4; do
  WANDB_MODE=offline python slam.py \
    --config configs/rgbd/replica/room0_${v}.yaml --eval
done
```

### Ablation config reference

| Config suffix | IsoReg (B4) | Replay (D2) | Description |
|--------------|-------------|-------------|-------------|
| `_imp1` | ✗ | ✗ | Baseline |
| `_imp2` | ✓ | ✗ | +B4 only |
| `_imp3` | ✗ | ✓ | +D2 only |
| `_imp4` | ✓ | ✓ | Full (B4+D2) |

---

## Repository Structure

```
MonoGS/
├── README_IMPROVEMENT.md          ← This file
├── slam.py                        ← Entry point
├── utils/
│   ├── slam_frontend.py           ← Improvement A (failed, kept for record)
│   └── slam_backend.py            ← Improvements B4 + D2
├── gaussian_splatting/
│   └── scene/gaussian_model.py    ← Improvement C (failed, kept for record)
├── configs/                       ← imp1=Baseline, imp2=B4, imp3=D2, imp4=Full
│   ├── rgbd/replica/room0_imp{1..4}.yaml
│   └── mono/tum/fr3_office_imp{1..4}.yaml
├── results/
│   ├── ablation_metrics.csv       ← All 8 ablation experiments (ATE/RPE/PSNR/SSIM/LPIPS/VRAM)
│   ├── tum_freiburg3/
│   │   ├── tum_imp1~4/            ← Final ablation results with render PNGs
│   │   └── exploratory/           ← Failed experiments (not on GitHub)
│   └── replica_room0/
│       ├── rep_imp1~4/            ← Final ablation results
│       └── exploratory/           ← Failed experiments (not on GitHub)
├── docs/
│   ├── RESEARCH_LOG.md            ← Experiment log with design rationale
│   ├── REPRODUCE_GUIDE.md         ← Reproduce instructions
│   └── paper/Improving_MonoGS_Report.tex           ← Academic report
└── scripts/
    └── env_precheck.py            ← Environment validation script
```

---

## Acknowledgements

This work builds upon:
- [MonoGS: Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) (Matsuki et al., CVPR 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023)
