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

```
# trainer.py:281
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
```

- `nbs` (nominal batch size) = **64**（default.yaml:110）
- 如果 `batch=8`，则 `accumulate = 64/8 = 8`
- 意味着：**每 8 个 batch 才执行一次 `optimizer.step()`**
- **等效 batch size 始终保持 64**，小 batch 也能稳定训练

### 2.2 优化器自动选择

```
# trainer.py:1003
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

- 当 `optimizer='auto'` 且总迭代次数 > 10000 时，**自动选择 MuSGD**
- VisDrone (6471张图, batch=8, epochs=200) ≈ 16万迭代 >> 1万，**自动用 MuSGD**
- MuSGD 是 YOLO26 专用优化器（SGD + Muon 混合）

### 2.3 学习率策略

```
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

### 2.4 End-to-End 训练机制（dual label assignment）

YOLO26 默认 `end2end=True`（yolo26.yaml:8），训练时同时使用 **one2many** 和 **one2one** 两个检测头，推理时只用 one2one 头实现无 NMS 推理。

#### 2.4.1 模型结构

```python
# head.py:111-113
if end2end:
    self.one2one_cv2 = copy.deepcopy(self.cv2)   # one2one 的 box head
    self.one2one_cv3 = copy.deepcopy(self.cv3)   # one2one 的 cls head
```

`end2end=True` 时，Detect 头会额外 `deepcopy` 一份 `cv2/cv3` 作为 one2one 头。原来的 `cv2/cv3` 保留作为 **one2many 头**，参数量几乎翻倍。

#### 2.4.2 前向传播

```python
# head.py:150-154
preds = self.forward_head(x, **self.one2many)          # one2many 头正常前向
if self.end2end:
    x_detach = [xi.detach() for xi in x]               # ⚠️ 关键：detach feature
    one2one = self.forward_head(x_detach, **self.one2one)
    preds = {"one2many": preds, "one2one": one2one}     # 同时输出两个头的预测
```

- one2one 头的输入是 **detach 后的 feature**，梯度不会回传到 backbone
- backbone **只由 one2many 头的梯度来优化**，one2one 头只更新自身参数
- 设计原因：one2many 要求"多个位置高响应"（NMS 范式），one2one 要求"仅一个位置高响应"（无 NMS 范式），两者对 backbone 的要求矛盾，detach 避免梯度冲突

#### 2.4.3 损失计算

```python
# loss.py:1157-1189
class E2ELoss:
    self.one2many = loss_fn(model, tal_topk=10)                    # topk=10（一对多分配）
    self.one2one  = loss_fn(model, tal_topk=7, tal_topk2=1)        # topk=7, topk2=1（一对一分配）

    # 总 loss
    loss = loss_one2many * o2m + loss_one2one * o2o
```

#### 2.4.4 权重衰减策略

```python
# loss.py:1164-1189
# 初始: o2m=0.8, o2o=0.2
# 最终: o2m=0.1, o2o=0.9

def decay(self, x):
    return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (0.8 - 0.1) + 0.1
```

o2m 权重从 0.8 **线性衰减**到 0.1，o2o 从 0.2 增长到 0.9。每个 epoch 结束时调用 `criterion.update()`（trainer.py:508-509）。

| 训练阶段 | o2m | o2o | 说明 |
|----------|-----|-----|------|
| 早期 | 0.8 | 0.2 | one2many 主导，backbone 获得丰富监督信号 |
| 后期 | 0.1 | 0.9 | one2one 头精细调优，学会一对一预测 |

> ⚠️ **早停注意**：衰减基于**预设的 epochs 总量**，不是实际训练 epoch。例如设 epochs=120 但 50 epoch 早停，o2o 只到 0.48（远没到 0.9）。如果在意 end2end 推理质量，建议 epochs 设为实际预期训练轮数，而非偏大上限。

#### 2.4.5 推理与验证

```python
# head.py:150-160 forward()
preds = self.forward_head(x, **self.one2many)
if self.end2end:
    x_detach = [xi.detach() for xi in x]
    one2one = self.forward_head(x_detach, **self.one2one)
    preds = {"one2many": preds, "one2one": one2one}
if self.training:
    return preds                                          # 训练：返回两个头
y = self._inference(preds["one2one"] if self.end2end else preds)  # 推理：只用 one2one
if self.end2end:
    y = self.postprocess(y.permute(0, 2, 1))              # topk，无需 NMS
return y if self.export else (y, preds)
```

`model.eval()` 后 `self.training=False`，走推理分支，**验证时用的是 one2one 头**。这意味着：

- 训练每轮验证的 mAP 基于 one2one 头计算
- 训练早期 one2one 头还很弱，验证 mAP 可能比 one2many 头差不少
- 随着 o2o 权重增长，one2one 头逐渐追上

> ⚠️ 如果早停导致 o2o 权重不够高，one2one 头可能未训练充分，验证 mAP 偏低。

#### 2.4.6 导出（fuse）

```python
# head.py:249-251 导出/fuse 时删除 one2many 头
def fuse(self):
    self.cv2 = self.cv3 = None  # 删除 one2many 头
```

#### 2.4.7 总结

| 阶段 | one2many | one2one |
|------|----------|---------|
| 训练 | ✅ 使用，梯度回传到 backbone | ✅ 使用，输入 detach，只更新自身参数 |
| 训练 loss 权重 | 0.8 → 0.1（衰减） | 0.2 → 0.9（增长） |
| 验证 | ❌ 不使用 | ✅ 使用，topk 无需 NMS |
| 推理/导出 | ❌ 丢弃（fuse 删除） | ✅ 使用，无需 NMS |

#### 2.4.7 GFLOPs 显示说明

模型初始化时 `self.info()` 不传 `imgsz`，默认用 640 计算 GFLOPs（见 `model_info()` 和 `get_flops()` 的默认参数）。因此不同 `imgsz` 训练时显示的 GFLOPs 相同，实际 GFLOPs 与 imgsz 成平方关系：

- 实际 GFLOPs ≈ 显示值 × (实际 imgsz / 640)²

---

## 三、关键参数详解

### 3.1 multi_scale（多尺度训练）

```
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
| `cache` | `False` | 图片缓存（详见 3.4） |
| `resume` | `False` | 续训 |

### 3.4 数据增强详解

#### 3.4.1 Mosaic 增强（默认启用）

Mosaic 是 Ultralytics 框架默认启用的数据增强方法，通过拼接四张图片来增加训练样本的多样性。

**原理**：
- 随机选择一张图片作为中心图片
- 从数据集中随机选择另外三张图片，按一定比例拼接填充剩余区域
- 由于 VisDrone 数据集中小目标较多，Mosaic 增强可以有效增加小目标的可见性

**操作流程**：
1. 随机生成拼接点坐标（中心点）
2. 将当前图片放置在四个象限中的一个
3. 随机选择其他三张图片填充剩余三个象限
4. 根据设定的缩放比例调整每张图片的大小
5. 将四张图片的信息合并为一张图片

**可视化示例**：
- 训练开始时会在输出目录生成 `train_batch0.jpg` 等文件
- 这些文件展示了经过 Mosaic 增强后的图片及标注框

**VisDrone 优势**：
- 小目标物体在拼接过程中更不容易丢失
- 增加了不同场景的混合，提高模型泛化能力
- 有助于模型学习更丰富的背景信息

#### 3.4.2 其他默认增强策略

| 增强类型 | 参数 | 说明 |
|----------|------|------|
| **RandAugment** | `auto_augment='randaugment'` | 随机应用多种增强变换组合 |
| **HSV颜色增强** | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` | 调整色调、饱和度、亮度 |
| **水平翻转** | `fliplr=0.5` | 50% 概率水平翻转 |
| **随机擦除** | `erasing=0.4` | 40% 概率随机擦除部分区域 |
| **几何变换** | `translate=0.1, scale=0.5` | 随机平移和缩放 |

### 3.5 cache（图片缓存）

```
# base.py:136
self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
```

| 参数值 | 效果 | 速度 | 可复现 | 说明 |
|--------|------|------|--------|------|
| `False` / 不设 | 每 epoch 重新读原图 | 最慢 | ✅ | 无额外开销 |
| `True` / `'ram'` | 缓存到内存 | 最快 | ❌ | `cache=True` 等价于 `cache='ram'` |
| `'disk'` | 缓存为 `.npy` 到磁盘 | 较快 | ✅ | 推荐折中方案 |

**为什么 `ram` 不可复现？**
- `cache='ram'` 用 `ThreadPool` 多线程并发解码图片，线程完成时序不确定
- 多线程可能导致 OpenCV JPEG 解码的浮点截断有微小差异，存入内存的图片数值略有不同
- `cache='disk'` 将图片序列化为 `.npy` 文件，后续 `np.load()` 读取，纯确定性

**cache 加速原理**：
- 无 cache：每 epoch → 磁盘读文件 → OpenCV 解码 → resize
- `cache='ram'`：首次加载后存内存，后续 epoch 直接内存读取，省掉磁盘 I/O + 解码
- `cache='disk'`：首次训练写 `.npy` 到数据集目录，后续 epoch 读 `.npy`，省掉 JPEG 解码

> **建议**：磁盘空间够用 `cache='disk'`（可复现 + 较快），内存充足不介意不可复现用 `cache='ram'`（最快）。

### 3.5 多卡训练

```
# trainer.py:270
batch_size = self.batch_size // max(self.world_size, 1)
```

`batch` 参数设的是**总 batch size**，多卡时自动平分到每张卡：

| device | batch | 每卡分到 | 等效总 batch |
|--------|-------|---------|-------------|
| `'0'` | 8 | 8 | 8 |
| `'0,1'` | 8 | 4 | 8 |
| `'0,1,2,3'` | 8 | 2 | 8 |

> 想每卡跑 8，需设 `batch=16`（双卡）或 `batch=32`（四卡）。

---

## 四、输出目录与实验管理

### 4.1 目录计算逻辑

```
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

```
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

```
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
    cache='disk',        # 缓存到磁盘（推荐），或 'ram'（更快但不可复现）
    pretrained=True,     # COCO 预训练

    # 实验管理
    project='visdrone',
    name=exp_name,
    exist_ok=True,
)
```

### 5.2 高分辨率配置（显存 ≥12GB）

```
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

> 数据来源：`docs/en/guides/yolo26-training-recipe.md`（YOLO26 官方训练 recipe 文档）

| Setting | N | S | M | L | X |
|---------|---|---|---|---|---|
| `epochs` | 245 | 70 | 80 | 60 | 40 |
| `lr0` | 0.0054 | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
| `lrf` | 0.0495 | 0.882 | 0.882 | 0.882 | 0.882 |
| `momentum` | 0.947 | 0.948 | 0.948 | 0.948 | 0.948 |
| `weight_decay` | 0.00064 | 0.00027 | 0.00027 | 0.00027 | 0.00027 |
| `warmup_epochs` | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| `batch` | 128 | 128 | 128 | 128 | 128 |
| `imgsz` | 640 | 640 | 640 | 640 | 640 |

> **注意**：COCO 预训练时大模型收敛更快（X 只需 40 epochs），但 VisDrone 只有 6k 张图，需要更多 epochs。N 模型用了更高的初始学习率 + 陡衰减（lrf=0.0495），S/M/L/X 用低学习率 + 温和衰减（lrf=0.882），反映了小模型需要更激进的更新。

#### 5.3.1 Recipe vs Default：两套参数体系的区别

| | default.yaml（微调场景） | COCO recipe（预训练场景） |
|---|---|---|
| **适用** | 小数据集微调 / 从零训练 | COCO 118k 大数据集预训练 |
| `lr0` | 0.01 | 0.00038（S/M/L/X） |
| `lrf` | 0.01 | 0.882 |
| 最终 lr | 0.01 × 0.01 = **0.0001** | 0.00038 × 0.882 = **0.000335** |
| lr 衰减幅度 | **衰减 100 倍**（0.01→0.0001） | **几乎不衰减**（0.00038→0.000335） |

**为什么 recipe 的 lr0 这么低、lrf 接近 1？**

COCO 预训练 = 118k 张图 × 128 batch × 40-245 epochs = 海量迭代。迭代次数巨大时：
- 初始 lr 必须很低（0.00038），否则训练爆炸
- lrf≈1 不衰减，因为 lr0 本身已经很低，不需要再大幅衰减

**为什么 default.yaml 的 lr0=0.01、lrf=0.01？**

微调场景数据少、迭代少，需要：
- 较高初始 lr（0.01）让模型快速适应新数据
- 大幅衰减（lrf=0.01）到极小值，最终精细收敛

**VisDrone 微调应该用哪个？**

用 default.yaml 的参数（`lr0=0.01, lrf=0.01`），即 `train.py` 当前配置。5.3 节的 recipe 是参考用的，让你了解预训练权重是怎么训出来的，**微调时不应照搬这些参数**。

---

## 六、Resume 续训指南

### 6.1 正确做法（推荐）

```
# 第一次训练：直接设目标 epochs，用 patience 早停
results = model.train(data='VisDrone.yaml', epochs=200, patience=50, ...)

# 如果早停了，想继续训练：
model = YOLO('runs/detect/visdrone/vis-26n-XXXXXXXX-XXXXXX/weights/last.pt')
results = model.train(resume=True)  # 继续原来的 200 轮计划
```

### 6.2 错误做法（避免）

```
# ❌ 不要中途改 epochs！
model.train(epochs=50)      # 先跑50轮
model.train(resume=True, epochs=100)  # 改成100轮，学习率曲线断裂！
```

---

## 七、训练日志与监控

### 7.1 日志文件

用 `train.sh` 启动训练，日志保存到带时间戳的文件：

```
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

默认 TQDM 每个 batch 都打印进度，日志量很大。可修改 `trainer.py` 和 `validator.py` 设置 `mininterval`：

```
# trainer.py:408，将 TQDM 添加 mininterval 参数
pbar = TQDM(enumerate(self.train_loader), total=nb, mininterval=60.0)
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

---

## 九、YOLO26 源码详解

### 9.1 模型架构

> 待补充：backbone、neck、head 结构详解，P2 小目标检测层等。

### 9.2 训练流程

> 待补充：trainer 训练循环、loss 计算、梯度累积等。

### 9.3 验证与推理

> 待补充：validator 验证流程、postprocess、NMS vs topk 等。

---

*最后更新：2026-04-13*
