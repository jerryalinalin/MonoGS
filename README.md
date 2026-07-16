> [!IMPORTANT]
> 本仓库基于 MonoGS 开展复现与方法改进。我的改进代码、消融实验、实验记录和复现说明位于 [`improvement`](../../tree/improvement) 分支；详细内容见 [`README_IMPROVEMENT.md`](https://github.com/jerryalinalin/MonoGS/blob/improvement/README_IMPROVEMENT.md)。下方内容为原始 MonoGS 项目说明

[comment]: <> (# Gaussian Splatting SLAM)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> Gaussian Splatting SLAM
  </h1>
  <p align="center">
    <a href="https://muskie82.github.io/"><strong>*Hidenobu Matsuki</strong></a>
    ·
    <a href="https://rmurai.co.uk/"><strong>*Riku Murai</strong></a>
    ·
    <a href="https://www.imperial.ac.uk/people/p.kelly/"><strong>Paul H.J. Kelly</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

  <h3 align="center"> CVPR 2024 (Highlight)</h3>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2312.06741">Paper</a> | <a href="https://youtu.be/x604ghp9R_Q?si=nYoWr8h2Xh-6L_KN">Video</a> | <a href="https://rmurai.co.uk/projects/GaussianSplattingSLAM/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
  </a>
  <a href="">
    <img src="./media/gui.jpg" alt="gui" width="100%">
  </a>
</p>
<p align="center">
This software implements dense SLAM system presented in our paper <a href="https://arxiv.org/abs/2312.06741">Gaussian Splatting SLAM</a> in CVPR'24.
The method demonstrates the first monocular SLAM solely based on 3D Gaussian Splatting (left), which also supports Stereo/RGB-D inputs (middle/right).
</p>
<br>

**Improvement branch**: This repo has been forked and improved as part of a course project. See [`README_IMPROVEMENT.md`](README_IMPROVEMENT.md) for details.

# Note
- In an academic paper, please refer to our work as **Gaussian Splatting SLAM** or **MonoGS** for short (this repo's name) to avoid confusion with other works.
- Differential Gaussian Rasteriser with camera pose gradient computation is available [here](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git).
- **[New]** Speed-up version of our code is available in `dev.speedup` branch.

# Getting Started
## Installation
```
git clone https://github.com/muskie82/MonoGS.git --recursive
cd MonoGS
```
Setup the environment.

```
conda env create -f environment.yml
conda activate MonoGS
```

Our test setup were:
- Ubuntu 20.04: `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6`
- Ubuntu 18.04: `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3`

## Quick Demo
```
bash scripts/download_tum.sh
python slam.py --config configs/mono/tum/fr3_office.yaml
```

## Run
### Monocular
python slam.py --config configs/mono/tum/fr3_office.yaml

### RGB-D
python slam.py --config configs/rgbd/replica/room0.yaml

## Evaluation
To evaluate our method, please add `--eval`:
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```

# Acknowledgement
This work incorporates many open-source codes.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Open3D](https://github.com/isl-org/Open3D)
- [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

# Citation
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{Gaussian} {Splatting} {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
