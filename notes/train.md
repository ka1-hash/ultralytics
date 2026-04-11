# YOLO26 VisDrone 训练指南

## 一、YOLO26 模型对比

### 1.1 模型规模与性能

| Model | size (pixels) | mAP<sup>val 50-95</sup> | mAP<sup>val 50-95(e2e)</sup> | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | params (M) | FLOPs (B) |
|-------|---------------|------------------------|------------------------------|---------------------|--------------------------|-----------|-----------|
| **YOLO26n** | 640 | 40.9 | 40.1 | 38.9 ± 0.7 | 1.7 ± 0.0 | **2.4** | **5.4** |
| **YOLO26s** | 640 | 48.6 | 47.8 | 87.2 ± 0.9 | 2.5 ± 0.0 | **9.5** | **20.7** |
| **YOLO26m** | 640 | 53.1 | 52.5 | 220.0 ± 1.4 | 4.7 ± 0.1 | **20.4** | **68.2** |
| **YOLO26l** | 640 | 55.0 | 54.4 | 286.2 ± 2.0 | 6.2 ± 0.2 | **24.8** | **86.4** |
| **YOLO26x** | 640 | 57.5 | 56.9 | 525.8 ± 4.0 | 11.8 ± 0.2 | **55.7** | **193.9** |

### 1.2 模型选择建议

| Model | 适用场景 | Batch Size 建议 | VisDrone 推荐度 |
|-------|---------|----------------|----------------|
| YOLO26n | 边缘设备、移动端、CPU实时 | 64-128 | ⭐⭐ 精度偏低 |
| YOLO26s | 速度与精度平衡 | 32-64 | ⭐⭐⭐ 可尝试 |
| **YOLO26m** | 较高精度，中等计算资源 | 16-32 | ⭐⭐⭐⭐⭐ **推荐** |
| **YOLO26l** | GPU可用时追求高精度 | 8-16 | ⭐⭐⭐⭐⭐ **小目标更优** |
| YOLO26x | 最高精度，服务器部署 | 4-8 | ⭐⭐⭐⭐ 显存要求高 |

> **VisDrone 建议**：小目标密集场景，推荐 `yolo26m` 或 `yolo26l`。显存 ≥8GB 建议直接上 `yolo26l`。

---

## 二、核心训练机制

### 2.1 自动梯度累积

```python
# trainer.py:281
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
```

- `nbs` (nominal batch size) = **64**（default.yaml:110）
- 如果 `batch=8`，则 `accumulate = 64/8 = 8`
- 意味着：**每 8 个 batch 才执行一次 `optimizer.step()`**
- **等效 batch size 始终保持 64**，小 batch 也能稳定训练

### 2.2 优化器自动选择

```python
# trainer.py:1003
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

- 当 `optimizer='auto'` 且总迭代次数 > 10000 时，**自动选择 MuSGD**
- VisDrone (6471张图, batch=8, epochs=200) ≈ 16万迭代 >> 1万，**自动用 MuSGD**
- MuSGD 是 YOLO26 专用优化器（SGD + Muon 混合）

### 2.3 学习率策略

```python
# trainer.py:248-254
def _setup_scheduler(self):
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->lrf
    else:
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
```

| 策略 | 参数 | 说明 |
|------|------|------|
| **Linear**（默认） | `cos_lr=False` | 从 `lr0` 线性降到 `lr0 * lrf` |
| **Cosine** | `cos_lr=True` | 余弦衰减，resume 时曲线会断裂 |

> ⚠️ **注意**：如果用 cosine，中途 resume 改 epochs 会破坏学习率曲线。建议直接设目标 epochs + patience 早停。

---

## 三、关键参数详解

### 3.1 multi_scale（多尺度训练）

```yaml
multi_scale: 0.0  # 0.0=禁用, 0.5=启用
```

- **作用**：训练时随机缩放图片尺寸
- **计算**：`imgsz * (1 ± multi_scale)`，然后取 stride 的倍数
- **举例**：`imgsz=640, multi_scale=0.5` → 随机从 **320~960** 之间选尺寸

**VisDrone 注意事项**：
- 小目标极多（行人、自行车），缩放到 320 时可能消失
- **建议**：`multi_scale=0.0` 或保守的 `0.2~0.3`

### 3.2 默认启用参数（default.yaml）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `save` | `True` | 自动保存检查点 |
| `pretrained` | `True` | 使用 COCO 预训练权重 |
| `amp` | `True` | 混合精度训练（加速） |
| `deterministic` | `True` | 确定性操作（可复现） |

### 3.3 默认关闭参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cos_lr` | `False` | 余弦学习率（默认 Linear） |
| `cache` | `False` | 图片载入内存（内存够建议开） |
| `resume` | `False` | 续训 |

---

## 四、输出目录与实验管理

### 4.1 目录计算逻辑

```python
# cfg/__init__.py: get_save_dir()
project = args.project or ""                      # 例如 'visdrone'
project = RUNS_DIR / args.task / project          # runs/detect/visdrone
save_dir = project / name                         # runs/detect/visdrone/<name>
```

| 参数 | 作用 | 示例 |
|------|------|------|
| `project` | 项目子目录名 | `'visdrone'` |
| `name` | 实验名称（即 run-id） | `'vis-26n-20260411-213824'` |
| `exist_ok` | 目录已存在时是否覆盖 | `True` 覆盖 / `False` 报错 |

**最终路径**：`runs/detect/<project>/<name>/`

> ⚠️ 不要设 `project='runs/detect'`，会导致路径变成 `runs/detect/runs/detect/<name>/`（重复）。

### 4.2 用时间戳作为 run-id

用时间戳命名实验，避免自动生成 `train`, `train1`, `train2`... 方便区分每次训练：

```python
from datetime import datetime

exp_name = f"vis-26n-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
# 例如：vis-26n-20260411-213824

results = model.train(
    ...
    project='visdrone',
    name=exp_name,
    exist_ok=True,    # 同名目录覆盖（时间戳基本不会重复）
)
```

### 4.3 训练输出目录内容

以 `runs/detect/visdrone/vis-26n-20260411-213824/` 为例：

**训练开始时生成：**

| 文件 | 说明 |
|------|------|
| `args.yaml` | 训练配置存档（所有参数），用于复现或 resume |
| `labels.jpg` | 数据集标签可视化（类别分布、框大小/位置统计） |
| `train_batch0.jpg` | 第 0 个 batch 经数据增强后的图像和标注 |
| `train_batch1.jpg` | 第 1 个 batch |
| `train_batch2.jpg` | 第 2 个 batch |

**训练过程中/完成后生成：**

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | ✅ **验证指标最好的权重**（部署用这个） |
| `weights/last.pt` | 最后一轮训练的权重（resume 用这个） |
| `weights/epochN.pt` | 设置 `save_period=N` 时，每 N 轮保存 |
| `results.csv` | 每轮训练的 loss、mAP 等指标数据 |
| `results.png` | 训练曲线图（loss, mAP 随 epoch 变化） |
| `confusion_matrix.png` | 混淆矩阵 |
| `F1_curve.png` | F1-Confidence 曲线 |
| `PR_curve.png` | Precision-Recall 曲线 |
| `P_curve.png` | Precision-Confidence 曲线 |
| `R_curve.png` | Recall-Confidence 曲线 |
| `val_batch0_pred.jpg` | 验证集预测结果可视化 |
| `val_batch0_labels.jpg` | 验证集标签可视化 |

---

## 五、VisDrone 推荐配置

### 5.1 基础配置（推荐）

```python
from datetime import datetime
from ultralytics import YOLO

model = YOLO('yolo26m.pt')
exp_name = f"vis-26m-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

results = model.train(
    data='VisDrone.yaml',
    epochs=100,          # 先试100轮，效果不好再 resume 增加
    patience=30,         # 30轮不提升就早停
    imgsz=640,           # 小目标多可尝试 1280（显存翻倍）
    batch=8,             # 显存够可改16

    # 优化器（自动选 MuSGD，可显式指定）
    optimizer='auto',    # 或 'MuSGD'
    lr0=0.01,            # MuSGD 默认 0.01

    # 数据增强（VisDrone 小目标多，保守设置）
    multi_scale=0.0,     # 禁用或 0.2-0.3
    copy_paste=0.2,      # 复制粘贴小目标

    # 其他
    cache='ram',         # 载入内存加速（内存够）
    pretrained=True,     # COCO 预训练

    # 实验管理
    project='visdrone',
    name=exp_name,
    exist_ok=True,
)
```

### 5.2 高分辨率配置（显存 ≥12GB）

```python
model = YOLO('yolo26l.pt')
exp_name = f"vis-26l-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

results = model.train(
    data='VisDrone.yaml',
    epochs=100,
    patience=30,
    imgsz=1280,          # 小目标更清晰
    batch=4,             # 显存翻倍，batch 减半
    multi_scale=0.0,     # 固定分辨率
    optimizer='MuSGD',
    lr0=0.01,
    project='visdrone',
    name=exp_name,
    exist_ok=True,
)
```

### 5.3 COCO 预训练参考参数

| Setting | N | S | M | L | X |
|---------|---|---|---|---|---|
| `epochs` | 245 | 70 | 80 | 60 | 40 |
| `lr0` | 0.0054 | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
| `lrf` | 0.0495 | 0.882 | 0.882 | 0.882 | 0.882 |
| `batch` | 128 | 128 | 128 | 128 | 128 |

> **注意**：COCO 预训练时大模型收敛更快（X 只需 40 epochs），但 VisDrone 只有 6k 张图，需要更多 epochs。

---

## 六、Resume 续训指南

### 6.1 正确做法（推荐）

```python
# 第一次训练：直接设目标 epochs，用 patience 早停
results = model.train(data='VisDrone.yaml', epochs=200, patience=50, ...)

# 如果早停了，想继续训练：
model = YOLO('runs/detect/visdrone/vis-26n-XXXXXXXX-XXXXXX/weights/last.pt')
results = model.train(resume=True)  # 继续原来的 200 轮计划
```

### 6.2 错误做法（避免）

```python
# ❌ 不要中途改 epochs！
model.train(epochs=50)      # 先跑50轮
model.train(resume=True, epochs=100)  # 改成100轮，学习率曲线断裂！
```

---

## 七、训练日志与监控

### 7.1 日志文件

用 `train.sh` 启动训练，日志保存到带时间戳的文件：

```bash
#!/bin/bash
# 项目级配置，避免和其他项目冲突
export YOLO_CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)/.config"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

# 生成带时间戳的日志文件名，和训练目录时间戳对应
LOG_FILE="logs/vis-26n-$(date +%Y%m%d-%H%M%S).log"
CUDA_VISIBLE_DEVICES=3 python train.py > "$LOG_FILE" 2>&1 &

echo "Training started, log: $LOG_FILE"
```

### 7.2 控制日志打印频率

默认 TQDM 每个 batch 都打印进度，日志量很大。可修改 `trainer.py` 设置 `mininterval`：

```python
# trainer.py:408，将 TQDM 添加 mininterval 参数
pbar = TQDM(enumerate(self.train_loader), total=nb, mininterval=10.0)
```

| mininterval | 效果 |
|-------------|------|
| `0.1`（默认） | 每个 batch 都打印，日志量大 |
| `60.0` | 每 60 秒打印一次 |

> ⚠️ **不要用** `os.environ['RUNPOD_POD_ID'] = '1'`，这会进入非交互模式导致训练进度完全不打印。

### 7.3 训练指标

| 文件/指标 | 位置 | 说明 |
|-----------|------|------|
| `results.csv` | `<save_dir>/results.csv` | 每轮的 box_loss, cls_loss, dfl_loss, mAP 等 |
| `metrics/mAP50(B)` | results.csv 中 | 验证集 mAP@0.5（核心指标） |
| `results.png` | `<save_dir>/results.png` | 训练曲线可视化 |

---

## 八、权重下载位置

默认下载到 `./weights/` 目录（Git 仓库根目录下）。

修改位置：
```bash
yolo settings weights_dir=/mnt/sda1/ultralytics/weights
```

---

*最后更新：2026-04-11*
