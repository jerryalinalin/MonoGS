# MonoGS Improvement — Reproduce Guide

> **职责**：操作手册——环境搭建、编译、运行命令、配置开关、消融实验流程。  
> 实验结果、缺陷机理、方案设计原理见 `RESEARCH_LOG.md`。  
> 交付物结构与最终结果见 `README_IMPROVEMENT.md`。

---

## 进度状态

| Phase | 状态 | 备注 |
|-------|------|------|
| Phase 0：环境搭建 | ✅ | Linux only |
| Phase 1：基线复现 | ✅ | TUM + Replica 均通过 |
| Phase 2：缺陷可视化 | ✅ | B + D 两处缺陷完成 |
| Phase 3：改进A | ❌ | 4方案全部失败，保留代码记录 |
| Phase 4：改进B（B4） | ❌ | 假设不成立：ATE未改善，LPIPS退化 |
| Phase 5：改进C | ❌ | 2方案全部失败，保留代码记录 |
| Phase 6：改进D（D2） | ✅ | 10%历史帧重放，全指标改善 |
| Phase 7：消融实验 | ✅ | 8次实验完成 |
| Phase 8：GitHub + 报告 | 🔲 | 待整理 |

---

## Phase 0：环境搭建

> ⚠️ **Linux 专用**。Windows 不可用（multiprocessing fork/spawn 不兼容，详见 `RESEARCH_LOG.md §Phase 0`）。

### 0.1 克隆与分支

```bash
git clone https://github.com/jerryalinalin/MonoGS.git --recursive
cd MonoGS
git checkout improvement   # 所有改动均在此分支
```

### 0.2 Conda 环境

```bash
conda create -n MonoGS python=3.10 -y
conda activate MonoGS

# PyTorch 2.2.2 + CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 主要依赖（如遇内部源缺包，加 --index-url https://pypi.org/simple/）
pip install munch trimesh open3d imgviz PyOpenGL glfw PyGLM \
  wandb lpips rich ninja \
  --index-url https://pypi.org/simple/

# 版本敏感依赖（顺序重要）
pip install \
  "evo==1.11.0" \
  "torchmetrics<1.0" \
  "opencv-python<4.9" \
  "plyfile<1.0" \
  --index-url https://pypi.org/simple/

# 最后统一锁定（防止上述依赖的依赖拉升版本）
pip install "numpy<2" "setuptools<68" "matplotlib<3.6" \
  --index-url https://pypi.org/simple/
```

> ⚠️ **安装顺序**：numpy/setuptools/matplotlib 必须最后安装并锁定。任何其他包可能将 numpy 拉升至 2.x，导致运行时崩溃。

### 0.3 编译 CUDA 扩展

```bash
# 前置：确认 g++-12 已安装
apt-get install -y g++-12

# ⚠️ 必须先打补丁再编译 simple-knn
# 在 submodules/simple-knn/simple_knn.cu 第1行插入：
echo '#include <cfloat>' | cat - submodules/simple-knn/simple_knn.cu \
  > /tmp/patched.cu && mv /tmp/patched.cu submodules/simple-knn/simple_knn.cu
# 或手动编辑添加该行

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

### 0.4 环境验证（必须全部 PASS）

```bash
python scripts/env_precheck.py
```

预检脚本覆盖：

| 检查项 | 预期 |
|--------|------|
| numpy 版本 | < 2.0 |
| evo 版本 | == 1.11.0 |
| matplotlib 版本 | < 3.6 |
| setuptools 版本 | < 68 |
| opencv-python 版本 | < 4.9 |
| plyfile 版本 | < 1.0 |
| pkg_resources 可导入 | ✓ |
| evo + matplotlib 联动画图 | ✓（触发 colorbar，验证兼容性） |
| diff-gaussian-rasterization 前向传播 | ✓（RMurai fork，5返回值） |
| GPU 残留进程 | 无（若有则打印 kill 命令） |

> 全部 PASS 后才运行 slam.py。

### 0.5 数据集

```bash
mkdir -p datasets/replica datasets/tum

# Replica room0（合成，RGB-D，~1.5 GB）
bash scripts/download_replica.sh
# 目标路径：datasets/replica/room0/

# TUM fr3/office（真实，单目，~13 GB）
bash scripts/download_tum.sh
# 目标路径：datasets/tum/rgbd_dataset_freiburg3_long_office_household/
```

验证路径：

```
datasets/
├── replica/room0/
└── tum/rgbd_dataset_freiburg3_long_office_household/
    ├── rgb/
    ├── depth/
    └── groundtruth.txt
```

---

## Phase 1：基线复现

### 代码结构速览

```
MonoGS/
├── slam.py                         # 入口，启动 Tracking/Mapping/GUI 三进程
├── utils/
│   ├── slam_frontend.py            # Tracking 线程 ← 改进A探索（失败）
│   └── slam_backend.py             # Mapping 线程 ← 改进B4 + 改进D2
├── gaussian_splatting/scene/
│   └── gaussian_model.py           # Gaussian 数据结构 ← 改进C探索（失败）
└── configs/
    ├── rgbd/replica/room0.yaml     # 场景A（RGB-D 原始配置）
    └── mono/tum/fr3_office.yaml    # 场景B（Monocular 原始配置）
```

### 随机种子固定

所有消融实验配置文件末尾均设置 `seed: 42`。已在以下位置追加种子固定代码：

```python
# slam.py SLAM.__init__ 和 slam_backend.py run() 均已添加：
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

### 运行基线

```bash
# 场景A：Replica room0（RGB-D）
WANDB_MODE=offline python slam.py \
  --config configs/rgbd/replica/room0_imp1.yaml --eval

# 场景B：TUM fr3/office（Monocular）
WANDB_MODE=offline python slam.py \
  --config configs/mono/tum/fr3_office_imp1.yaml --eval
```

### 论文对照参考值

| 指标 | Replica room0（论文） | TUM fr3/office（论文） |
|------|---------------------|---------------------|
| ATE RMSE ↓ (cm) | 0.44 | 3.50 |
| PSNR ↑ (dB) | 34.83 | 21.89 ★ |
| SSIM ↑ | 0.954 | 0.733 ★ |
| LPIPS ↓ | 0.068 | 0.327 ★ |

> ★ TUM 渲染指标为论文 Table 15a 的三序列平均值（fr1/desk + fr2/xyz + fr3/office），非 fr3/office 单独值。  
> 复现容忍范围：ATE ≤10%，渲染指标 ≤5%。

---

## Phase 2：缺陷可视化

追加**只读**可视化代码，不修改任何算法逻辑（全部包裹在 `torch.no_grad()` 中）。

### 缺陷B 可视化

| 追加位置 | 输出文件 | 内容 |
|---------|---------|------|
| `slam.py` evaluation 结束后 | `aniso_distribution.png` | 高斯各向异性比率直方图 |
| `slam.py` evaluation 结束后 | `aniso_stats.txt` | 平均拉伸比、>10×占比等统计数值 |

### 缺陷D 可视化

| 追加位置 | 输出文件 | 内容 |
|---------|---------|------|
| `slam_backend.py` `map()` 末尾 | `historical_frame_residual.png` | 窗口内帧 vs 历史帧平均残差双折线图 |
| `slam_backend.py` `map()` 末尾 | `historical_frame_residual.jsonl` | 原始数值（每次map()记录一条） |

运行命令（与基线相同）：

```bash
WANDB_MODE=offline python slam.py \
  --config configs/mono/tum/fr3_office_imp1.yaml --eval
```

---

## Phase 4：改进B — 视线感知各向同性正则化（B4）

**修改文件**：`utils/slam_backend.py`

### 配置开关

```yaml
# configs/mono/tum/fr3_office_imp2.yaml 中的 Training 节点
Training:
  iso_reg: True
  iso_reg_mode: "b4"         # "b1"=阈值截断, "b4"=视线方向感知（最优）
  iso_reg_lambda: 0.01
  seed: 42
```

### B4 核心代码（`utils/slam_backend.py`）

```python
def isotropic_loss_view_aware(gaussians, camera_pos, lambda_iso=0.01):
    """
    视线方向感知各向同性正则化。
    对齐视线方向的高斯拉伸是有害的（新视角渲染消失），
    垂直视线方向的拉伸是合理的（高效覆盖表面）。
    用视线对齐度对拉伸比惩罚项加权，使约束自动聚焦于有害拉伸。
    """
    if lambda_iso <= 0:
        return torch.tensor(0.0, device="cuda")

    scales = gaussians.get_scaling          # [N, 3]
    rotations = gaussians.get_rotation      # [N, 4] quaternion
    positions = gaussians.get_xyz           # [N, 3]
    N = scales.shape[0]

    # 最长轴方向（世界坐标系）
    R = build_rotation(rotations)           # [N, 3, 3]
    max_axis_idx = scales.argmax(dim=1)     # [N]
    longest_axis = R[torch.arange(N), :, max_axis_idx]  # [N, 3]
    longest_axis = F.normalize(longest_axis, dim=1)

    # 视线方向（高斯中心指向相机）
    viewing_dir = F.normalize(
        camera_pos.unsqueeze(0) - positions, dim=1
    )  # [N, 3]

    # 对齐度：1=平行于视线（有害），0=垂直于视线（合理）
    alignment = (longest_axis * viewing_dir).sum(dim=1).abs()  # [N]

    # 各向异性比率
    s_max = scales.max(dim=1).values
    s_min = scales.min(dim=1).values
    ratio = s_max / (s_min + 1e-6)

    # 视线方向加权惩罚
    loss = (alignment * torch.clamp(ratio - 1.0, min=0.0)).mean()
    return lambda_iso * loss
```

在 `map()` 中调用（追加在原始 loss 计算之后）：

```python
# 原始 loss（保持不变）
loss = (1.0 - opt_params.lambda_dssim) * Ll1 \
     + opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))

# 改进B4：视线感知各向同性正则化
if self.config["Training"].get("iso_reg", False) \
        and self.config["Training"].get("iso_reg_mode", "") == "b4":
    lambda_iso = self.config["Training"].get("iso_reg_lambda", 0.01)
    loss = loss + isotropic_loss_view_aware(
        self.gaussians,
        viewpoint_cam.camera_center,
        lambda_iso
    )
```

---

## Phase 6：改进D — 历史关键帧重放（D2）

**修改文件**：`utils/slam_backend.py`

### 配置开关

```yaml
# configs/mono/tum/fr3_office_imp3.yaml 中的 Training 节点
Training:
  keyframe_replay: True
  replay_freq: 10            # 每10次迭代插入1帧历史帧（占比10%）
  seed: 42
```

### D2 核心代码（`utils/slam_backend.py`）

在 `map()` 函数内，找到 viewpoint 采样部分，修改如下：

```python
# 当前窗口帧列表（保持不变）
viewpoint_stack = [self.cameras[i] for i in current_window]

# 改进D：历史帧池（窗口外的所有已记录关键帧）
replay_pool = [i for i in self.cameras.keys()
               if i not in set(current_window)]
replay_freq = self.config["Training"].get("replay_freq", 10)
use_replay  = self.config["Training"].get("keyframe_replay", False)

for iteration in range(mapping_iters):
    # 每隔 replay_freq 次，插入一帧历史帧
    if use_replay and len(replay_pool) > 0 \
            and iteration % replay_freq == 0:
        cam = self.cameras[random.choice(replay_pool)]
    else:
        cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
    
    # 以下 render → loss → backward 流程完全不变
    render_pkg = render(cam, self.gaussians,
                        self.pipeline_params, self.background)
    # ...
```

> ⚠️ 此修改不涉及 tracking 模块，不修改窗口 OC 逻辑，tracking 帧集合不变。

---

## Phase 7：消融实验

### 配置文件说明

8 份配置文件已在仓库中准备好，命名规则统一为 `*_imp{1-4}.yaml`：

| 后缀 | B4（IsoReg） | D2（Replay） | 说明 |
|------|------------|------------|------|
| `_imp1` | ✗ | ✗ | Baseline |
| `_imp2` | ✓ | ✗ | +B4 only |
| `_imp3` | ✗ | ✓ | +D2 only |
| `_imp4` | ✓ | ✓ | Full（B4+D2） |

Training 节点关键参数差异：

```yaml
# _imp1（Baseline）
Training:
  iso_reg: False
  keyframe_replay: False
  seed: 42

# _imp2（+B4）
Training:
  iso_reg: True
  iso_reg_mode: "b4"
  iso_reg_lambda: 0.01
  keyframe_replay: False
  seed: 42

# _imp3（+D2）
Training:
  iso_reg: False
  keyframe_replay: True
  replay_freq: 10
  seed: 42

# _imp4（Full）
Training:
  iso_reg: True
  iso_reg_mode: "b4"
  iso_reg_lambda: 0.01
  keyframe_replay: True
  replay_freq: 10
  seed: 42
```

### 执行命令

```bash
# Replica room0（RGB-D）× 4
for v in imp1 imp2 imp3 imp4; do
  echo "=== Running Replica $v ==="
  WANDB_MODE=offline python slam.py \
    --config configs/rgbd/replica/room0_${v}.yaml --eval
done

# TUM fr3/office（Monocular）× 4
for v in imp1 imp2 imp3 imp4; do
  echo "=== Running TUM $v ==="
  WANDB_MODE=offline python slam.py \
    --config configs/mono/tum/fr3_office_${v}.yaml --eval
done
```

> ⚠️ 每次实验结束后，将 `results/` 下最新时间戳目录中的 `metrics.json` 和渲染 PNG 立即归档到 `results/experiments/{版本}/{场景}/`，避免最后整理时混淆。

### 指标记录表（手动填入 `results/metrics.csv`）

```
scene,version,ATE,RPE,PSNR,SSIM,LPIPS,train_time_s,vram_gb
replica,baseline,,,,,,,
replica,+B4,,,,,,,
replica,+D2,,,,,,,
replica,Full,,,,,,,
tum,baseline,,,,,,,
tum,+B4,,,,,,,
tum,+D2,,,,,,,
tum,Full,,,,,,,
```

显存：训练中途执行 `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 记录峰值。  
训练时长：`slam.py` 末尾打印的 `Total time` 字段。

---

## Phase 8：GitHub 整理

### 最终仓库结构

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

### 定性对比图生成

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 从每个版本的结果目录中选取3个代表性场景的渲染图
scenes   = ['white_wall', 'desk', 'corridor']   # 选低纹理 + 中等纹理 + 走廊
versions = ['Baseline', '+B4', '+D2', 'Full']
base_dir = 'results/experiments'

fig, axes = plt.subplots(len(scenes), len(versions) + 1,
                          figsize=(20, 12))  # +1 列放 GT

for i, scene in enumerate(scenes):
    for j, ver in enumerate(versions):
        img_path = f'{base_dir}/{ver}/tum/{scene}_render.png'
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[i][j].imshow(img)
        axes[i][j].set_title(ver if i == 0 else '', fontsize=10)
        axes[i][j].set_ylabel(scene if j == 0 else '', fontsize=9)
        axes[i][j].axis('off')
    # GT 列
    gt_path = f'{base_dir}/baseline/tum/{scene}_gt.png'
    if os.path.exists(gt_path):
        axes[i][-1].imshow(mpimg.imread(gt_path))
    axes[i][-1].set_title('GT' if i == 0 else '', fontsize=10)
    axes[i][-1].axis('off')

plt.tight_layout()
plt.savefig('results/qualitative_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: results/qualitative_comparison.png')
```

### ZIP 交付物打包

```bash
# 课程要求：渲染结果PNG + 指标CSV/JSON
zip -r 学号_姓名_SLAM2026Final.zip \
  results/experiments/ \
  results/metrics.csv \
  results/qualitative_comparison.png
```

---

## 评分维度自查清单

提交前逐项确认：

- [ ] **A维度（15分）**：TUM ATE 偏差 −4.3%（✅）；Replica ATE 完美匹配（✅）；LPIPS 偏差原因已在报告中分析（AlexNet权重差异）
- [ ] **B维度（20分）**：缺陷B（各向异性）+ 缺陷D（历史帧零监督）机理独立（✅）；均有量化可视化证据；均超出原论文承认的局限（原论文承认"无回环"和"未达30fps"）
- [ ] **C维度（20分）**：改进B假设（ATE↓≥5%）和改进D假设（PSNR↑≥0.3dB，ATE变化≤5%）均为可证伪形式；与对应缺陷机理逻辑对应
- [ ] **D维度（25分）**：8次消融实验同硬件/同超参/同种子（seed=42）（✅）；消融表包含 ATE+RPE+PSNR+SSIM+LPIPS+训练时长（✅）；定性对比图≥3组
- [ ] **E维度（20分）**：IEEE 模板；Discussion 诚实分析改进A/C失败原因和 Full 版本 PSNR 下降原因；结论明确回答假设是否成立
- [ ] **GitHub**：`improvement` 分支（✅）；`README_IMPROVEMENT.md` 含一行复现命令（✅）；`results/metrics.csv` 有原始数据；`results/qualitative_comparison.png` 存在
- [ ] **AI 使用声明**：报告末尾包含 AI Usage Statement，注明 Claude 辅助范围
