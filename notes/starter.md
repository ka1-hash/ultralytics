# Ultralytics YOLO 项目入门指南

## 📋 项目概述

| 特性 | 说明 |
|------|------|
| **任务支持** | 目标检测、实例分割、姿态估计、图像分类、旋转目标检测 |
| **模型版本** | YOLOv3/v5/v8/v9/v10/11/26|
| **使用方式** | Python API + CLI 命令行 (`yolo` 命令) |
| **部署支持** | ONNX、TensorRT、OpenVINO、CoreML、TF Lite 等 |

---

## SAR目标检测（示例）

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # 加载 YOLOv8-medium 预训练模型
results = model.train(data='ssdd.yaml', epochs=12, imgsz=640, batch=8)
```

- **数据集**：SSDD（船舶检测数据集）
- **配置**：`ultralytics/cfg/datasets/ssdd.yaml`
- **运行**：通过 `train.sh` 在 GPU 3 上后台训练

---

## 📁 根目录文件说明

| 文件/目录 | 用途说明 |
|-----------|----------|
| `pyproject.toml` | **Python 项目配置**，定义依赖、版本、入口点、构建工具（uv/pip 安装用） |
| `uv.lock` | **uv 包管理器锁定文件**，记录精确依赖版本，确保环境可复现 |
| `mkdocs.yml` | **文档网站配置**，定义 https://docs.ultralytics.com 的结构和主题 |
| `README.md` / `README.zh-CN.md` | 项目介绍文档（英文/中文） |
| `LICENSE` | AGPL-3.0 开源许可证 |
| `CITATION.cff` | 学术引用格式文件，方便研究者引用本项目 |
| `CONTRIBUTING.md` | 贡献指南，说明如何提交 PR、代码规范等 |
| `.gitignore` / `.dockerignore` | Git 和 Docker 的忽略文件配置 |
| `train.py` / `train.sh` | **你的训练脚本**（自定义添加），用于启动训练 |
| `docker/` | Docker 镜像构建文件，支持多种 CUDA 版本环境 |
| `docs/` | **官方文档源文件**（MkDocs 格式），构建后生成静态网站 |
| `examples/` | 使用示例代码（Python、CLI、ROS、YOLOv5 迁移等） |
| `tests/` | 单元测试和集成测试代码 |
| `scripts/` | **你的辅助脚本**（自定义添加） |
| `logs/` | **训练日志输出目录**（自定义添加） |
| `notes/` | **你的笔记文档**（自定义添加），包含本入门指南 |
| `.github/` | GitHub Actions 工作流、ISSUE 模板等（详见下方） |
| `ultralytics.egg-info/` | pip 安装生成的包元数据（自动生成，无需关注） |

---

## 核心目录 ultralytics

### 1. `cfg/` - 配置文件目录

| 文件/目录 | 用途说明 |
|-----------|----------|
| `__init__.py` | 配置模块初始化，包含配置加载和解析函数 |
| `default.yaml` | **默认训练配置文件**，包含所有训练超参数（学习率、batch size、优化器等） |
| `datasets/` | 数据集配置文件目录，包含 COCO、VOC 等标准数据集配置 |
| `models/` | 模型架构 YAML 文件目录，定义各版本 YOLO 的网络结构 |
| `trackers/` | 跟踪器配置文件（botsort.yaml、bytetrack.yaml） |

#### `cfg/datasets/` - 数据集配置
- `coco.yaml`, `voc.yaml`, `imagenet.yaml` - 标准数据集配置
- `ssdd.yaml` - **SAR船舶检测数据集配置**

#### `cfg/models/` - 模型架构配置
| 目录 | 说明 |
|------|------|
| `v3/`, `v5/`, `v6/`, `v8/`, `v9/`, `v10/`, `11/`, `12/`, `26/` | 各版本 YOLO 模型配置 |
| `rt-detr/` | RT-DETR 模型配置（实时检测Transformer） |

---

### 2. `data/` - 数据处理模块

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 数据模块初始化 |
| `annotator.py` | 用yolo26+sam自动标注 |
| `augment.py` | **数据增强实现**（Mosaic、Mixup、随机翻转、HSV增强等） |
| `base.py` | 数据集基类定义 |
| `build.py` | 数据加载器构建函数 |
| `converter.py` | 数据集格式转换工具（COCO ↔ YOLO ↔ VOC 等） |
| `dataset.py` | **数据集类实现**（YOLODataset、ClassificationDataset 等） |
| `loaders.py` | 数据加载器（支持图片、视频、流、屏幕捕获等） |
| `split.py` | 数据集划分工具（训练/验证/测试集分割） |
| `split_dota.py` | DOTA 数据集专用分割工具（处理大图像） |
| `utils.py` | 数据处理工具函数 |

#### `data/explorer/` - 数据探索工具（已移除 ⚠️）

> **注意**：从 `ultralytics>=8.3.12` 开始，Explorer 功能已从代码库中移除，迁移至 [Ultralytics Platform](https://platform.ultralytics.com/)。

**原功能**：Explorer 是一个基于 LanceDB 向量数据库的数据集探索工具，支持：
- **语义搜索** - 用自然语言搜索相似图片（如"找包含狗的图片"）
- **SQL 查询** - 直接对数据集执行 SQL 查询
- **向量相似度搜索** - 基于图像嵌入找到相似样本
- **GUI 界面** - 浏览器可视化操作，命令 `yolo explorer` 启动

**移除原因**：
1. **维护成本** - 依赖 LanceDB、OpenAI 等外部服务，增加维护负担
2. **功能重叠** - 类似功能已集成到 Ultralytics Platform 云服务
3. **专注核心** - 团队希望专注于 YOLO 训练和推理核心功能
4. **使用率低** - 相比训练/推理功能，Explorer 使用场景相对小众

如需使用，可降级到 `pip install ultralytics==8.3.11` 或直接使用 Ultralytics Platform。

#### `data/scripts/` - 数据下载脚本
- `get_coco.sh`, `get_coco128.sh` - COCO 数据集下载
- `get_imagenet.sh` - ImageNet 下载
- `download_weights.sh` - 预训练权重下载

---

### 3. `engine/` - 核心引擎（**最重要！**）

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 引擎模块初始化 |
| `model.py` | **YOLO 模型主类**，封装训练、验证、预测、导出等所有功能 |
| `trainer.py` | **训练器实现**，包含完整的训练循环、优化器设置、学习率调度 |
| `validator.py` | **验证器实现**，模型评估和指标计算 |
| `predictor.py` | **预测器实现**，推理逻辑（支持批处理、流式处理） |
| `exporter.py` | **模型导出器**，支持 ONNX、TensorRT、OpenVINO、CoreML 等格式 |
| `tuner.py` | 超参数调优器（Ray Tune 集成） |
| `results.py` | 推理结果封装类（Boxes、Masks、Keypoints 等） |

---

### 4. `models/` - 模型定义（按任务类型组织）

| 文件/目录 | 用途说明 |
|-----------|----------|
| `__init__.py` | 模型模块初始化 |
| `fastsam/` | FastSAM 模型（快速分割一切） |
| `nas/` | NAS（神经架构搜索）模型 |
| `rtdetr/` | RT-DETR 实时检测Transformer |
| `sam/` | SAM（Segment Anything Model）分割模型 |
| `yolo/` | **YOLO 各任务实现** |
| `utils/` | 模型工具函数和损失函数 |

#### `models/yolo/` - YOLO 任务实现
| 目录 | 用途 |
|------|------|
| `model.py` | YOLO 模型类定义 |
| `detect/` | **目标检测**：train.py, val.py, predict.py |
| `segment/` | **实例分割**：train.py, val.py, predict.py |
| `pose/` | **姿态估计**：train.py, val.py, predict.py |
| `classify/` | **图像分类**：train.py, val.py, predict.py |
| `obb/` | **旋转目标检测**（Oriented Bounding Box） |
| `world/` | **开放词汇检测**（YOLO-World） |
| `yoloe/` | YOLO-E 模型实现 |

---

### 5. `nn/` - 神经网络模块

| 文件/目录 | 用途说明 |
|-----------|----------|
| `__init__.py` | 神经网络模块初始化 |
| `tasks.py` | **模型构建核心**，根据 YAML 配置构建检测/分割/姿态模型 |
| `autobackend.py` | **多后端推理支持**，自动选择 PyTorch/ONNX/TensorRT 等 |
| `text_model.py` | 文本模型（用于 YOLO-World 开放词汇检测） |

#### `nn/modules/` - 网络层模块
| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 模块初始化，导出所有层 |
| `conv.py` | 卷积层实现（Conv、DWConv、GhostConv 等） |
| `block.py` | **网络块实现**（C2f、SPPF、Bottleneck、ResBlock 等） |
| `head.py` | **检测头实现**（Detect、Segment、Pose、OBB 头） |
| `transformer.py` | Transformer 模块（注意力机制、多头注意力等） |
| `activation.py` | 激活函数（SiLU、Mish、Swish 等） |
| `utils.py` | 模块工具函数 |

#### `nn/backends/` - 推理后端实现
| 文件 | 用途 |
|------|------|
| `base.py` | 后端基类 |
| `pytorch.py` | PyTorch 后端 |
| `onnx.py` | ONNX Runtime 后端 |
| `tensorrt.py` | TensorRT 后端（NVIDIA GPU 加速） |
| `openvino.py` | OpenVINO 后端（Intel 加速） |
| `coreml.py` | CoreML 后端（Apple 设备） |
| `tensorflow.py` | TensorFlow/TFLite 后端 |
| `rknn.py`, `ncnn.py`, `mnn.py` | 移动端/嵌入式后端 |

---

### 6. `solutions/` - 应用解决方案（**开箱即用**）

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 解决方案模块初始化 |
| `solutions.py` | 解决方案基类 |
| `config.py` | 解决方案配置管理 |
| `object_counter.py` | **目标计数器**（支持区域计数、越线计数） |
| `object_tracking.py` | **目标跟踪**（带 ID 的跟踪） |
| `heatmap.py` | **热力图生成**（人流密度可视化） |
| `speed_estimation.py` | **速度估计**（车辆速度计算） |
| `distance_calculation.py` | **距离计算**（目标间距离） |
| `region_counter.py` | **区域计数**（多区域统计） |
| `queue_management.py` | **队列管理**（排队长度检测） |
| `parking_management.py` | **停车场管理**（车位检测） |
| `security_alarm.py` | **安全报警**（入侵检测） |
| `ai_gym.py` | **AI 健身**（动作计数、姿态评估） |
| `vision_eye.py` | **视线追踪** |
| `instance_segmentation.py` | **实例分割应用** |
| `object_blurrer.py` | **目标模糊**（隐私保护） |
| `object_cropper.py` | **目标裁剪**（提取检测目标） |
| `trackzone.py` | **区域跟踪** |
| `analytics.py` | **分析工具**（数据统计可视化） |
| `similarity_search.py` | **相似度搜索** |
| `streamlit_inference.py` | **Streamlit Web 界面** |

---

### 7. `trackers/` - 目标跟踪算法

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 跟踪器模块初始化 |
| `basetrack.py` | 跟踪器基类 |
| `byte_tracker.py` | **ByteTrack 算法实现** |
| `bot_sort.py` | **BoT-SORT 算法实现** |
| `track.py` | 跟踪状态管理（TrackState 类） |
| `README.md` | 跟踪器使用文档 |

---

### 8. `utils/` - 工具函数库

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 工具模块初始化，包含常用函数 |
| `loss.py` | **损失函数实现**（DetectionLoss、SegmentationLoss、PoseLoss 等） |
| `metrics.py` | **评估指标**（mAP计算、混淆矩阵、F1分数等） |
| `ops.py` | **核心操作**（NMS、框转换、坐标变换等） |
| `plotting.py` | **可视化工具**（画框、画掩码、结果保存等） |
| `torch_utils.py` | PyTorch 工具（模型信息、设备管理、分布式训练） |
| `checks.py` | 环境检查（依赖版本、CUDA 可用性等） |
| `downloads.py` | 文件下载工具 |
| `files.py` | 文件操作工具 |
| `instance.py` | 实例封装（Boxes、Masks、Keypoints、Probs） |
| `tal.py` | **任务对齐分配**（Task Aligned Assigner，标签分配策略） |
| `nms.py` | **非极大值抑制**实现 |
| `benchmarks.py` | 模型基准测试工具 |
| `autobatch.py` | 自动 batch size 计算 |
| `autodevice.py` | 自动设备选择（CPU/GPU/MPS） |
| `logger.py` | 日志记录（Comet、ClearML、TensorBoard） |
| `events.py` | 事件处理 |
| `tuner.py` | 超参数调优工具 |
| `triton.py` | NVIDIA Triton 推理服务器支持 |
| `dist.py` | 分布式训练工具 |
| `git.py` | **Git 仓库信息读取**（无需 git 命令），获取分支、commit SHA、origin 远程地址，用于训练日志记录版本信息 |
| `cpu.py` | CPU 优化工具 |
| `patches.py` | 补丁工具 |
| `tqdm.py` | 进度条工具 |
| `uploads.py` | 文件上传工具 |
| `errors.py` | 错误处理 |

---

### 9. `optim/` - 优化器实现

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 优化器模块初始化 |
| `muon.py` | **Muon 优化器** - 基于正交化的矩阵乘法子空间优化器，使用 Newton-Schulz 迭代进行梯度正交化处理 |

**Muon 优化器**：
- 一种新型优化器，专门优化矩阵乘法子空间
- 使用 Newton-Schulz 迭代替代 SVD 进行正交化，计算效率更高
- 适用于大规模神经网络训练，在某些场景下比 Adam 收敛更快

---

### 10. `hub/` - Ultralytics HUB 云服务集成

**Ultralytics HUB** 是官方提供的云端训练和模型管理平台（https://hub.ultralytics.com/），`hub/` 模块是本地代码与云平台的连接桥梁。

| 文件 | 用途说明 |
|------|----------|
| `__init__.py` | 模块入口，提供用户-facing API：login/logout、模型导出、数据集检查 |
| `auth.py` | **认证管理**，处理 API Key 验证、Google Colab 免密登录、请求头生成 |
| `session.py` | **训练会话管理**，本地训练时与云端同步指标、上传检查点、心跳保活 |
| `utils.py` | HUB 相关工具函数（API 根地址、请求封装、帮助信息） |
| `google/` | Google Colab 环境专用适配（自动获取 Colab 认证信息） |

**核心功能**：
- **登录认证** - `hub.login(API_KEY)` 绑定云端账号
- **云端训练** - 从 HUB 拉取模型配置，训练时实时同步指标到云端
- **模型管理** - 导出、重置、下载云端训练的模型
- **数据集检查** - 上传前验证数据集格式是否正确

---

### 10. `assets/` - 示例资源

- `bus.jpg`, `zidane.jpg` - 示例图片用于测试

---

### 11. 根目录文件

| 文件 | 用途 |
|------|------|
| `__init__.py` | Ultralytics 包初始化，定义版本和导出 |
| `py.typed` | PEP 561 类型标记文件，告诉类型检查器"这个包有类型信息，可以检查 |

---

## 🚀 推荐入手路径

### 第 1 步：跑通当前训练

你的 `train.py` 已经可以运行。训练完成后会生成 `runs/detect/train/` 目录，包含：
- 权重文件 (`best.pt`, `last.pt`)
- 训练曲线 (`results.png`)
- 验证样本 (`val_batch*.jpg`)

### 第 2 步：学习基础用法（先看 examples）

```bash
# 查看官方教程笔记本
jupyter notebook examples/tutorial.ipynb
```

关键示例：
| 文件 | 内容 |
|------|------|
| `tutorial.ipynb` | 完整使用教程 |
| `object_tracking.ipynb` | 目标跟踪 |
| `object_counting.ipynb` | 目标计数 |

### 第 3 步：尝试 CLI 命令

除了 Python API，还可以直接用命令行：

```bash
# 训练（和你当前的 train.py 等效）
yolo detect train data=ssdd.yaml model=yolov8m.pt epochs=12 imgsz=640 batch=8

# 验证
yolo detect val model=runs/detect/train/weights/best.pt data=ssdd.yaml

# 预测
yolo detect predict model=runs/detect/train/weights/best.pt source='image.jpg'

# 导出 ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### 第 4 步：深入理解（按需选择）

| 方向 | 查看文件 | 说明 |
|------|----------|------|
| **修改模型** | `ultralytics/nn/tasks.py` | 添加新模块、修改架构 |
| **自定义损失** | `ultralytics/utils/loss.py` | 修改损失函数 |
| **数据增强** | `ultralytics/data/augment.py` | 自定义增强策略 |
| **评估指标** | `ultralytics/utils/metrics.py` | 自定义评估方式 |
| **导出部署** | `ultralytics/engine/exporter.py` | 支持多种格式导出 |

---

## 📚 关键配置文件

1. **`ultralytics/cfg/default.yaml`** - 所有训练参数的默认值
2. **`ultralytics/cfg/datasets/ssdd.yaml`** - 你的数据集配置（已创建）
3. **`pyproject.toml`** - 项目依赖和元信息

---

### `.github/` 目录详解

GitHub 自动化配置目录，包含 CI/CD 工作流和交互模板。

#### 1. `dependabot.yml` - 依赖自动更新
- 自动检查并创建 PR 更新依赖包版本
- 监控 `pyproject.toml` 和 GitHub Actions 的更新

#### 2. `ISSUE_TEMPLATE/` - Issue 提交模板

| 文件 | 用途 |
|------|------|
| `bug-report.yml` | Bug 报告表单，引导用户提供复现步骤、环境信息等 |
| `feature-request.yml` | 功能请求表单，描述期望功能和用例 |
| `question.yml` | 问题咨询表单，引导用户先查文档 |
| `config.yml` | 模板配置，设置默认标签和禁止空 Issue |

#### 3. `workflows/` - GitHub Actions 工作流

| 文件 | 触发条件 | 用途 |
|------|----------|------|
| `ci.yml` | PR / push | **主 CI 流程**：运行测试、代码检查、多平台验证 |
| `format.yml` | PR | 自动代码格式化（Black、Ruff） |
| `docs.yml` | push / 定时 | 构建并部署文档网站 |
| `docker.yml` | push / 定时 | 构建并推送 Docker 镜像到 Docker Hub |
| `publish.yml` | release | 发布到 PyPI |
| `conda-check-prs.yml` | PR | 检查 Conda 环境兼容性 |
| `links.yml` | 定时 | 检查文档中的死链 |
| `stale.yml` | 定时 | 自动标记和关闭长期无活动的 Issue/PR |
| `cla.yml` | PR | 检查贡献者是否签署 CLA（贡献者许可协议） |
| `merge-main-into-prs.yml` | 定时 | 自动将 main 分支更新合并到 PR |
| `mirror.yml` | push | 代码镜像到其他平台（如 Gitee） |

---

## 💡 下一步建议

### 如果你想快速出成果：
- 训练完当前模型后，用 `model.predict()` 做推理测试
- 查看 `solutions/` 里的应用方案（计数、跟踪等）

### 如果你想做研究/改进：
- 阅读 `ultralytics/engine/trainer.py` 理解训练流程
- 从 `ultralytics/nn/tasks.py` 入手理解模型构建
- 修改 `ultralytics/data/augment.py` 尝试新的数据增强

### 如果你想部署：
- 用 `model.export(format='onnx')` 导出模型
- 查看 `examples/` 里的 C++/ONNX Runtime 示例
