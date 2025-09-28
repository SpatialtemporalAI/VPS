# VPS: Visual Positioning System

A coarse-to-fine visual positioning system that combines the power of Hierarchical-Localization (Hloc)  for visual place recognition and multiple pose estimation methods (VGGT, MASt3R) for accurate pose estimation.

## Features

- Efficient visual place recognition using [MegaLoc](https://github.com/gmberton/MegaLoc)
- Multiple pose estimation methods:
  - [VGGT](https://github.com/facebookresearch/vggt?tab=readme-ov-file)
  - [Pi3](https://github.com/yyfz/Pi3)
- Scale recovery：
  - use Motion Averaging(multi camera only)
  - use input ground truth depth by rgbd camera information 
  - use [moge](https://github.com/microsoft/moge) to Monocular Depth   Estimation
  - use ref database depth and rgb
- Easy-to-use pipeline for visual localization
- Configurable pose estimation methods

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SpatialtemporalAI/VPS.git
cd VPS
```

2. Create a conda environment:
```bash
conda create -n vps python=3.10
conda activate vps
pip install scikit-image joblib flask
```

3. Install third_party dependencies:
```bash
mkdir third_party && cd third_party

#3.1 VGGT
git clone https://github.com/facebookresearch/vggt.git
cd vggt 
#  pytorch with cuda
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
cd ..

#3.2 hloc
git clone https://github.com/Benaidandan/Hierarchical-Localization.git
cd Hierarchical-Localization/
python -m pip install -e .
cd ..

#3.3 (Optional) Pi3
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -r requirements.txt #change torch with cuda
cd ..

##3.4(Optional) depth pred
pip install git+https://github.com/microsoft/MoGe.git

```
4. (Optional) Download models
```bash
mkdir checkpoints && cd checkpoints

```
- [vggt_1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)
- [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal)


## Usage
### Basic Usage
- input: rgb + depth(Optional)
- ref database：rgb+pose(c2w) 
- output:  
  - 4x4 pose c2w(txt /data/outputs/pose)
  - server json: {'theta': , 'x': , 'y': }

```bash
#start server 
python service.py 
#test cilent
python scripts/test_client.py --url xxx --image xxx --depth xxx
```

## Structure
### Project Structure
```
VPS/
├── vps/                   # Main source code
│   ├── core/              # Core modules
│   │   ├── pose_vggt.py   # VGGT pose estimator
│   │   ├── pose_pi3.py    # Pi3 pose estimator
│   │   ├── vpr.py         # Visual place recognition
│   │   ├── depth_pred.py  # Depth pred（m）
│   │   └── __init__.py    # Main VPS class
│   ├── utils/             # tool
├── configs/               # Configuration files
│   ├── default.yaml       # Default config (VGGT) 重要
│   └── data.yaml          # 配合scripts/colmap_to_vps.py将colmap转需要的格式，不用管
├── log/                   # logger
├── service.py             # Dataset directory
```
### Data Structure
```
├── data/                  # Dataset directory
│   ├── ref/               # ref database 参考数据库(来源是slam 或者colmap的) 下面默认是文件夹名字
│   │   ├── rgb/            # png jpg
│   │   ├── poses/          # c2w  4x4 txt
│   │   ├── depth/          # 深度 npy  单位m (Optional，没有的话启用moge单目深度估计 在d455效果不错)
│   │   ├── depth_render/   # 渲染的深度 npy  单位m (Optional)
│   │   ├── calibration/    # Camera Intrinsics 3x3 txt(Optional 暂时没用到)
│   │   ├── rgb_render/     # rgb rendered by mesh or 3d Gaussian
│   │
│   ├── outputs/           # 
│   │   ├── poses/          #vps result dir c2w  4x4 txt
│   │   ├── temp/          
│   │   ├── pairs.txt      #vpr result
│   │   ├── query.h5       #
│   │   ├── ref.h5         #
│   │   ├── result.txt     #eval
│   │   ├── last_pose.txt  #last vps result
│   │
│   ├── query/           #service.py和 scripts.py共用
数据结构如上就行 与config中的default.yaml相对应即可
在data中rgb，depth,啥的除了尾缀格式不一样外，要求same filename 
```