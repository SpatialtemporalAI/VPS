# VPS: Visual Positioning System

A coarse-to-fine visual positioning system that combines the power of Hierarchical-Localization (Hloc)  for visual place recognition and multiple pose estimation methods (VGGT, MASt3R) for accurate pose estimation.

## Features

- Efficient visual place recognition using [MegaLoc](https://github.com/gmberton/MegaLoc) in [Hloc](https://github.com/cvg/Hierarchical-Localization)
- Multiple pose estimation methods:
  - accurate pose estimation
  - [VGGT](https://github.com/facebookresearch/vggt?tab=readme-ov-file)
  - [MASt3R](https://github.com/naver/mast3r)
- Scale recovery：
  - use input ground truth depth by rgbd camera information 
  - use [moge](https://github.com/microsoft/moge) to Monocular Depth Estimation  
  - use ref database depth and rgb
- Easy-to-use pipeline for visual localization
- Configurable pose estimation methods

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/Benaidandan/VPS.git
cd VPS
```

2. Create a conda environment:
```bash
conda create -n vps python=3.10
conda activate vps
pip install -r requirements.txt
```

3. Install dependencies:
```bash
#vggt
cd third_party/vggt
# install pytorch 2.3.1 with cuda 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
cd ..
cd Hierarchical-Localization/
python -m pip install -e .
cd ..
#For MASt3R support, refer to https://github.com/naver/mast3r?tab=readme-ov-file#installation
##
cd mast3r
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
pip install -r dust3r/requirements_optional.txt
pip install cython
pip install faiss-gpu
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .  # or python3 setup.py build_ext --inplace
cd ../../..
## depth pred
pip install git+https://github.com/microsoft/MoGe.git

```
4. (Optional) Download models
```bash
mkdir checkpoints
cd checkpoints
```
- [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal)
- [vggt_1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)

## Usage
### Basic Usage
- input: rgb + depth(可选)
- ref数据库：rgb+pose(c2w) 相机到世界
- output:  
  - 4x4 pose c2w(txt格式放到/data/outputs/pose下)
  - 服务器返回json: {'theta': , 'x': , 'y': }

```bash
#启动服务器
python service.py 
#模拟测试服务器
#python scripts/test_client.py --url xxx --image xxx --depth xxx
```

## Structure
### Project Structure
```
VPS/
├── vps/                   # Main source code
│   ├── core/              # Core modules
│   │   ├── pose.py        # VGGT pose estimator
│   │   ├── pose_mast3r.py # MASt3R pose estimator
│   │   ├── vpr.py         # Visual place recognition
│   │   ├── depth_pred.py  # Depth pred （m）
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
│   │   ├── last_pose.txt  #last vps result（并没有节省时间）
│   │
│   ├── query/           #service.py和 scripts.py共用
数据结构如上就行 与config中的default.yaml相对应即可
在data中rgb，depth,啥的除了尾缀格式不一样外，要求same filename 
```