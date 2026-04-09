# VPS: Visual Positioning System

一个基于粗到精两阶段流程的视觉定位服务：

- 第一阶段用 VPR 做候选参考图检索
- 第二阶段用 pose model 做相机位姿估计
- 服务端支持多地图与多机器人会话
- 当前主运行链路已经从旧的 `vps/core` 大类实现，解耦为 `manager / maps / session / pipeline / models`

## 当前状态

当前服务启动和请求处理走的是新架构：

```text
service.py
  -> LocalizationCoordinator
    -> MapManager / SessionManager / ModelManager
    -> VPRPipeline
    -> PosePipeline
    -> VPRModel / PoseModel
```

`vps/core` 目录里的代码仍然保留，主要用于历史实现和部分兼容，但 `service.py` 的主路径已经不再直接依赖旧的大一统 `VisualPositioningSystem`。

## 主要特性

- 基于 HLoc/MegaLoc 的视觉地点检索
- 解耦的 VPR 与 Pose 两阶段流水线
- 支持多地图注册与预加载
- 支持多机器人独立会话
- 支持按 `robot_id` 串行化请求，避免同机器人高频并发时覆盖 query 文件
- 支持可选的深度导航后处理，返回 `occupancy_map`
- 服务端可返回 2D 位姿：
  - `x`
  - `y`
  - `theta`

## 安装

### 1. 克隆仓库

```bash
git clone git@github.com:SpatialtemporalAI/VPS.git
cd VPS
```

### 2. 创建环境

```bash
conda create -n vps python=3.10
conda activate vps
pip install scikit-image joblib flask
```

### 3. 安装第三方依赖

```bash
mkdir third_party && cd third_party

# 3.1 VGGT
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
cd ..

# 3.2 HLoc
git clone https://github.com/Benaidandan/Hierarchical-Localization.git
cd Hierarchical-Localization
python -m pip install -e .
cd ..

# 3.3 Optional: Pi3
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -r requirements.txt
cd ..

# 3.4 Optional: depth prediction
pip install git+https://github.com/microsoft/MoGe.git
```

### 4. 可选：下载模型权重

```bash
mkdir checkpoints && cd checkpoints
```

- [vggt_1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)
- [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal)

## 快速开始

### 启动服务

```bash
python service.py
```

默认读取配置文件：

```text
configs/default.yaml
```

### 发送一次定位请求

可以直接用当前仓库里的模拟脚本：

```bash
python a.py --image /path/to/query.jpg
```

带机器人和地图：

```bash
python a.py \
  --image /path/to/query.jpg \
  --robot-id 1 \
  --map-id map_b
```

带深度：

```bash
python a.py \
  --image /path/to/query.jpg \
  --depth /path/to/query.npy \
  --robot-id 0
```

### `/localize` 请求字段

服务接口为：

```text
POST /localize
```

`multipart/form-data` 字段如下：

- `image`: 必填，查询 RGB 图像，支持 `.jpg/.jpeg/.png`
- `robot_id`: 可选，机器人 ID，默认是 `0`
- `map_id`: 可选，目标地图 ID
- `depth`: 可选，深度文件，支持：
  - `.png`，单位毫米，服务端会转成 `.npy`
  - `.npy`，单位米

返回 JSON：

- `pose`: `{"x", "y", "theta"}` 或 `null`
- `map_png_b64`: 可选，PNG 编码后的结果图
- `map_format`: 有图时为 `"png"`，否则为 `null`

## 启动流程

`service.py` 启动后会按以下顺序创建对象：

1. 读取 `configs/default.yaml`
2. 创建：
   - `ModelManager`
   - `MapManager`
   - `SessionManager`
3. 创建模型：
   - `VPRModel`
   - `PoseModel`
4. 创建 pipeline：
   - `VPRPipeline`
   - `PosePipeline`
5. 创建 `LocalizationCoordinator`
6. 从 `service.maps` 解析地图配置
7. 对每张地图执行：
   - `register_map(...)`
   - `prepare_map(...)`
8. 将默认机器人 `robot_id=0` 绑定到第一张地图

## 请求处理流程

一次 `/localize` 请求的主流程如下：

1. 服务端接收 `image / depth / robot_id / map_id`
2. 为该 `robot_id` 获取专属请求锁
3. 将 query 图保存到 `service.temp_dir`
4. 根据 `robot_id` 和 `map_id` 绑定当前活动地图
5. `LocalizationCoordinator.localize(...)`
6. `VPRPipeline.run(...)` 从当前地图中检索参考图
7. `PosePipeline.run(...)` 估计位姿并做 motion averaging
8. 若开启 `depth_nav`，尝试生成 `occupancy_map`
9. 返回 2D 位姿和可选地图结果

## 解耦后的目录结构

```text
VPS/
├── service.py
├── a.py
├── configs/
│   └── default.yaml
├── vps/
│   ├── core/                # 旧实现，保留作兼容/迁移参考
│   ├── manager/             # 运行时管理层
│   │   ├── localization_coordinator.py
│   │   ├── map_manager.py
│   │   ├── model_manager.py
│   │   └── session_manager.py
│   ├── maps/                # 静态地图数据封装
│   │   ├── map_instance.py
│   │   ├── pose_map.py
│   │   └── vpr_map.py
│   ├── models/              # 模型后端封装
│   │   ├── pose_model.py
│   │   ├── pose_model_contract.py
│   │   ├── vggt_model.py
│   │   └── vpr_model.py
│   ├── pipeline/            # 业务流水线
│   │   ├── pose_pipeline.py
│   │   └── vpr_pipeline.py
│   ├── session/             # 机器人会话状态
│   │   └── robot_session.py
│   ├── nav/                 # 地图更新/投影相关逻辑
│   └── utils/
├── data/
└── log/
```

## 各层职责

### `models/`

只负责模型初始化与原始推理，不负责地图管理和业务调度。

- `VPRModel`
  - 根据配置加载 VPR 后端
  - 提取全局 descriptor
- `PoseModel`
  - 根据 `pose.method` 选择具体 pose backend
  - 当前服务主路径已迁入 `VGGT`
- `pose_model_contract.py`
  - 定义 pose model 的统一输入输出约束

### `maps/`

只负责地图静态数据和查找接口。

- `VPRMap`
  - 参考图路径
  - 参考图 descriptor
  - 参考图 pose tensor
- `PoseMap`
  - 参考图 pose 映射
  - 深度、render、标定、nav 地图等静态资源路径
- `MapInstance`
  - 把同一张地图的 `VPRMap` 和 `PoseMap` 绑定在一起

### `pipeline/`

只负责业务流程，不直接管理多机器人状态。

- `VPRPipeline`
  - query descriptor 提取
  - 相似度检索
  - 可选空间过滤
- `PosePipeline`
  - pose model 推理
  - motion averaging
  - query depth 提取
  - 可选 depth navigation

### `manager/`

负责运行时编排。

- `ModelManager`
  - 管理已注册模型实例
- `MapManager`
  - 管理地图注册、预加载、descriptor/pose 入内存
- `SessionManager`
  - 管理不同 `robot_id` 的会话状态
- `LocalizationCoordinator`
  - 串联 map/session/model/pipeline 整个定位流程

### `session/`

- `RobotSession`
  - `active_map_id`
  - `last_pose`

## 配置说明

当前地图配置统一放在：

```yaml
service:
  maps:
    - id: map_a
      ref_data_path: /path/to/map_a
      ref_descriptors_path: /path/to/ref_a.h5
      depth_dir: /path/to/map_a/depth                # optional
      calibration_dir: /path/to/map_a/calibration    # optional
      render_rgb_dir: /path/to/map_a/rgb_render      # optional
      render_depth_dir: /path/to/map_a/depth_render  # optional
      nav_map_path: /path/to/map_a/map.png           # optional
      nav_yaml_path: /path/to/map_a/map.yaml         # optional
```

注意：

- `service.maps` 是当前新服务的地图入口
- `ref_data_path` 下默认使用：
  - `rgb/`
  - `poses/`
- `nav_map_path` 和 `nav_yaml_path` 需要按地图配置
- 如果不配置 nav 底图资源，则不会生成导航相关地图结果

## 数据组织

推荐的单地图目录结构：

```text
map_a/
├── rgb/             # 参考图像
├── poses/           # 每张参考图对应的 4x4 c2w txt
├── depth/           # 可选，深度 npy，单位 m
├── calibration/     # 可选，相机内参 3x3 txt
├── rgb_render/      # 可选，render RGB
└── depth_render/    # 可选，render depth
```

要求：

- `rgb/` 和 `poses/` 通过同名文件 stem 对齐
- 例如：
  - `rgb/000123.jpg`
  - `poses/000123.txt`

## 多机器人与地图切换

- 默认机器人 ID 是 `0`
- 服务启动时，默认机器人会绑定到 `service.maps` 的第一张地图
- 新机器人如果第一次请求想定位到非默认地图，需要在请求里显式带上 `map_id`
- 同一个 `robot_id` 的请求会串行执行，避免覆盖同名 query 文件
- 不同 `robot_id` 之间可以并发请求

## 当前已迁移与未迁移部分

### 已迁移到新架构

- 服务启动装配
- 地图注册与预加载
- 多机器人 session
- VPR pipeline
- VGGT pose pipeline
- motion averaging
- 可选 depth navigation

### 仍保留在旧实现或待迁移

- `vps/core` 下的历史 pose/VPR 代码
- 某些旧评测脚本仍可能引用旧入口
- 非 VGGT 的 pose backend 仍在逐步迁移中

## 备注

- `README` 描述的是当前 `service.py` 主运行路径
- 如果你在看 `vps/core`，请把它理解为旧实现和迁移参考，而不是当前主入口
