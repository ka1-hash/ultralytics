# ReCon 类别平衡完整方案说明（终版）

## 1. 方案定位

本文档只聚焦 **VisDrone 三尾类类别平衡**，对象为：

- 6：tricycle
- 7：awning-tricycle
- 8：bus

本方案当前**不把“小目标放大 / chip 放大 / P2-P4 结构改造”作为主变量**。这些问题可以在后续单独作为第二阶段继续做，但本方案的核心问题是：

> 如何基于真实高质量实例、语义合理位置、ReCon 受控生成与多重质检，稳定生成 6/7/8 三类的新样本，使类别分布更合理，同时不把原始训练分布做坏。

本方案最终采用：

```text
真实高质量实例
→ SAM 提取纯目标
→ 候选位置生成与语义评分
→ 设计 target_bbox
→ ReCon 局部生成 / 修复
→ Grounding DINO + SAM + YOLO26 质检
→ 小规模验证
→ 全量构建与训练
```

## 2. 最终判断

### 2.1 三类都做，不再只做 bus

旧实验里，bus 相对更稳，awning-tricycle 更容易掉点。但这不应被解释为“只有 bus 可以增强”，更合理的解释是：

1. 旧 copy / paste / blend 方案的样本质量不够；
2. 6/7 类对位置语义、边界质量、背景纯净度更敏感；
3. 不是 6/7 不能增强，而是旧增强样本不够真、不够纯、不够稳。

所以本方案改成：

- **6 / 7 / 8 三类全部纳入生成主线**；
- 但三类生成风险不同，因此 **阈值、通过条件、扩量速度不同**。

### 2.2 ReCon 的准确定位

ReCon 在本方案里不是“更高级的复制粘贴器”，而是：

1. **给定 bbox 的局部受控生成器**；
2. **目标边界与背景过渡修复器**；
3. **目标语义纠偏器**；
4. **高质量新样本生产模块**。

它不负责：

- 自动发现所有最优位置；
- 自动替代检测器；
- 直接生成整图；
- 无限制增加尾类数量。

### 2.3 最核心的设计原则

1. **先把样本做对，再决定生成多少**；
2. **先小规模验证，再全量训练**；
3. **原始数据集不复制，只保存最终 pass 图像与标签**；
4. **所有规则都分为：硬规则、初始规则、统计驱动规则**；
5. **所有阈值都要回答两个问题：怎么计算、为什么这样设、最终如何校准。**

---

## 3. 模型功能与规则映射

### 3.1 SAM 负责什么

SAM 在本方案中承担四个功能：

1. **纯目标实例提取**：从 GT bbox 中提取目标 mask，而不是直接用 bbox 矩形块；
2. **候选位置空白检查**：判断该位置是否真的可放；
3. **生成后轮廓验收**：判断目标轮廓是否完整、是否碎裂；
4. **最终 bbox 更新**：当生成后的目标边界与初始框轻微偏移时，用 mask 外接框更新最终框。

### 3.2 Grounding DINO 负责什么

Grounding DINO 在本方案中承担四个功能：

1. **候选位置语义匹配**：这里像不像车该出现的位置；
2. **生成后类别一致性检查**：新目标是不是对应类别；
3. **多余目标检查**：patch 内是否多生了别的目标；
4. **负语义位置剔除**：roof/tree/water 等明显错误位置直接剔除。

### 3.3 ReCon / ControlNet 负责什么

ReCon 负责：

1. 在给定 `target_bbox` 内生成目标；
2. 修复边界、阴影、纹理过渡；
3. 在局部 patch 范围内进行多版本生成；
4. 在语义和几何约束下做受控生成，而不是自由生成。

### 3.4 YOLO26 baseline 负责什么

YOLO26 baseline 不是只用于最终训练，也参与样本治理：

1. 检查新样本是否可学习；
2. 找出“YOLO26 暂时检不出、但 GDINO/SAM 已通过”的高价值难例；
3. 过滤明显错类样本；
4. 过滤生成后引入的大量背景误检。

---

## 4. 规则分层

### 4.1 H 类：硬规则

这类规则原则上不建议放松：

- H1：val 集永远不增强；
- H2：原始数据集不复制；
- H3：只有通过 GDINO + SAM + YOLO26 三重质检的样本才允许进入训练；
- H4：patch / bbox 必须完整在图内；
- H5：与已有 GT 不允许重叠；
- H6：先 mini，再 full；
- H7：训练集使用 `raw + recon_pass` 联合清单，不重新复制整份数据集；
- H8：awning-tricycle 在同一张增强图中最多 1 个目标。

### 4.2 I 类：有真实依据的初始规则

这些规则来源于当前已统计出的尺寸分布、旧实验现象与工程保守值。它们有依据，但仍允许后续校准：

- S2：源实例 short_train 区间；
- S3：源实例质量评分 Q_src；
- S4：SAM 分割通过门槛；
- P1：候选位置评分 S_pos 初始阈值；
- Q1：Q_final 初始阈值。

### 4.3 T 类：统计驱动规则

这类规则必须先统计，再 mini 验证，最后锁死：

- C1：每类最终新增比例；
- C2：每张增强图允许新增几个目标；
- C3：每个源实例最多使用几次；
- Q_src 的最终阈值；
- SAM 指标的最终阈值；
- Q_final 的最终阈值。

---

## 5. 源实例规则（S 组）

## S1：源实例必须来自真实 GT

来源只允许：

- 原始 train 中的 6/7/8 类 GT；
- 不允许从旧增强图反向提取；
- 不允许从纯生成图反向提取。

目标：保证 ReCon 的源样本始终 anchored 在真实数据分布里。

## S2：实例尺寸下限

统一定义：

```text
short_train = min(box_w, box_h) * 1280 / image_long_side
```

这里不以小目标放大为主变量，但源样本本身必须可用，因此要先过滤掉过小、过不稳的源实例。

### S2 初始推荐区间

| 类别 | short_train 初始区间 | 依据 |
|---|---:|---|
| tricycle | 16–56 | 下限略高于该类 P25=15；上限取 P90 附近并略放松 |
| awning-tricycle | 20–56 | 分布与 tricycle 相近，但类别更脆弱，因此下限更高 |
| bus | 24–160 | 下限高于 P25=20，避开偏小不稳样本；上限宽放以保证源池规模 |

### 为什么这样设

1. **tricycle**：P25=15，P50=24。下限 16 的意思不是“16 最优”，而是先排除 P25 附近以下那批最不稳定的弱实例。
2. **awning-tricycle**：虽然尺寸分布与 tricycle 接近，但历史增强里它最容易掉点，所以同样大小下必须更严格，因此下限抬到 20。
3. **bus**：bus 本身分布更大、旧增强更稳定，因此下限设为 24 以避开偏小不稳样本，上限宽放到 160 以保留充足的高质量 bus 源池。

### S2 最终校准方式

S2 不是永久固定值。最终要结合：

- 不同下限对应的源池保留率；
- 人工抽查质量；
- mini 验证效果。

#### S2 统计命令

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
from collections import defaultdict

ROOT = Path('/home/shukang/datasets/VisDrone')
LBL = ROOT / 'labels/train'
IMG = ROOT / 'images/train'

targets = {6:'tricycle', 7:'awning-tricycle', 8:'bus'}
vals = defaultdict(list)

for p in sorted(LBL.glob('*.txt')):
    img_path = IMG / (p.stem + '.jpg')
    if not img_path.exists():
        img_path = IMG / (p.stem + '.png')
    if not img_path.exists():
        continue
    w, h = Image.open(img_path).size
    long_side = max(w, h)

    for line in p.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = map(float, line.split()[:5])
        cls = int(cls)
        if cls not in targets:
            continue
        short_train = min(bw*w, bh*h) * 1280.0 / long_side
        vals[cls].append(short_train)

for cls, name in targets.items():
    arr = sorted(vals[cls])
    print(f'\n== {name} ==')
    for lo in [12,14,16,18,20,24]:
        hi = 56 if cls in [6,7] else 160
        kept = sum(lo <= x <= hi for x in arr)
        print(f'lower={lo}: keep={kept}/{len(arr)} ({kept/len(arr):.3f})')
PY
```

最终要看：

- keep 数量是否足够；
- 被保留下来的样本是不是更干净；
- mini 训练后增益是否更稳定。

## S3：实例质量评分 Q_src

```text
Q_src = 0.25 * Q_size
      + 0.20 * Q_mask
      + 0.20 * Q_edge
      + 0.15 * Q_isolation
      + 0.10 * Q_boundary
      + 0.10 * Q_domain
```

### 它是什么

Q_src 不是自然统计量，而是一个 **工程化的源实例可用性评分**。它回答的问题是：

> 这个真实实例，适不适合拿来做 ReCon 生成的源样本？

### 每一项怎么得到

#### Q_size
按照 S2 区间定义一个中心化分数：

```text
若 short_train 不在 [L_c, U_c] 内，Q_size = 0
否则：Q_size = 1 - |short_train - M_c| / ((U_c - L_c)/2)
```

其中 `M_c` 是区间中点。

#### Q_mask
来自 SAM 结果：

```text
Q_mask = 0.5 * IoU(mask_bbox, gt_bbox)
       + 0.3 * area_score
       + 0.2 * cc_score
```

- `area_score`：mask_area / bbox_area 是否接近合理区间中部；
- `cc_score`：连通域越少、最大连通域占比越高，得分越高。

#### Q_edge
在 mask 边界 band 上计算梯度强度，并做分位归一化。

#### Q_isolation
统计外扩邻域中的目标数量和最大相邻 IoU：

```text
Q_isolation = 1 - 0.6 * min(neighbor_count / K, 1)
                - 0.4 * min(max_neighbor_iou / T, 1)
```

例如 `K=3`, `T=0.3`。

#### Q_boundary
源目标到图像边界的最小距离归一化：

```text
Q_boundary = clip(d_min / B, 0, 1)
```

可取 `B=16` 或 `24`。

#### Q_domain
这里不是物理一致性，而是“局部域稳定性”。可简单理解为：

- 亮度是否正常；
- 对比度是否正常；
- 是否更接近道路 / 交通目标常见背景；
- 是否来自高质量原图而不是异常背景。

### 为什么这么加权

- `Q_size / Q_mask / Q_edge` 决定“像不像一个干净、可迁移的目标源”，所以权重最高；
- `Q_isolation` 次之；
- `Q_boundary / Q_domain` 是辅助项。

### Q_src 初始阈值

- tricycle：`Q_src >= 0.65`
- awning-tricycle：`Q_src >= 0.75`
- bus：`Q_src >= 0.60`

原因：
- awning 最脆弱，因此阈值最高；
- bus 最稳，因此阈值最低；
- tricycle 居中。

### S3 最终校准方式

建议最终写成“初始阈值 + 分位数兜底”：

- tricycle：`Q_src >= max(0.65, P60_of_Qsrc_6)`
- awning：`Q_src >= max(0.75, P75_of_Qsrc_7)`
- bus：`Q_src >= max(0.60, P50_of_Qsrc_8)`

### S3 统计命令

```bash
python - <<'PY'
import numpy as np
import pandas as pd
csv_path = '/home/shukang/datasets/VisDrone_ReCon_Workspace/metadata/source_instances_raw.csv'
df = pd.read_csv(csv_path)
for cls, name in [(6,'tricycle'), (7,'awning'), (8,'bus')]:
    sub = df[df['class_id'] == cls]
    print(f'\n== {name} ==')
    for col in ['Q_size','Q_mask','Q_edge','Q_isolation','Q_boundary','Q_domain','Q_src']:
        arr = sub[col].dropna().values
        print(col,
              'p10', np.percentile(arr,10),
              'p25', np.percentile(arr,25),
              'p50', np.percentile(arr,50),
              'p75', np.percentile(arr,75),
              'p90', np.percentile(arr,90))
PY
```

最终要看：

- 各项分数分布；
- 总分分布；
- 高分样本是否肉眼也确实更干净。

## S4：SAM 分割通过条件

### 指标定义

1. `mask_bbox 与 GT_bbox IoU`
   - 用 SAM 得到二值 mask；
   - 取 mask 最小外接矩形为 `mask_bbox`；
   - 与 `GT_bbox` 计算标准 IoU。

2. `mask_area / bbox_area`
   - `mask_area` 为前景像素面积；
   - `bbox_area` 为 GT 框面积；
   - 比值太小表示只分到了一小块，太大表示吞入背景太多。

3. `最大连通域占比`
   - `largest_cc_ratio = max(ai) / sum(ai)`
   - 越接近 1 越说明 mask 主体完整。

4. `连通域数量`
   - 标准 connected components 数量。

### S4 初始通过条件

| 指标 | tricycle | awning | bus |
|---|---:|---:|---:|
| mask_bbox 与 GT_bbox IoU | >=0.60 | >=0.65 | >=0.60 |
| mask_area / bbox_area | 0.25–0.90 | 0.30–0.85 | 0.35–0.95 |
| 最大连通域占比 | >=0.70 | >=0.75 | >=0.75 |
| 连通域数量 | <=2 | <=2 | <=3 |

### 为什么这样设

- awning 需要更完整轮廓和车篷结构，所以 IoU、最大连通域占比都更严格；
- bus 允许最多 3 个连通域，因为长条大目标在遮挡或窗区干扰下更容易产生轻微分裂；
- tricycle 介于两者之间。

### S4 最终校准方式

最终要做：

1. 对全训练集三类跑 SAM；
2. 统计四个指标分布；
3. 人工抽查“好样本 / 差样本”；
4. 把阈值设在“好坏分布的分界附近”。

#### S4 统计命令

```bash
python - <<'PY'
import numpy as np
import pandas as pd
csv_path = '/home/shukang/datasets/VisDrone_ReCon_Workspace/metadata/source_instances_sam.csv'
df = pd.read_csv(csv_path)
for cls, name in [(6,'tricycle'), (7,'awning'), (8,'bus')]:
    sub = df[df['class_id'] == cls]
    print(f'\n== {name} ==')
    for col in ['mask_bbox_iou', 'mask_area_ratio', 'largest_cc_ratio', 'cc_count']:
        arr = sub[col].dropna().values
        if len(arr) == 0:
            continue
        print(col,
              'p10', np.percentile(arr,10),
              'p25', np.percentile(arr,25),
              'p50', np.percentile(arr,50),
              'p75', np.percentile(arr,75),
              'p90', np.percentile(arr,90))
PY
```

---

## 6. 位置规则（P 组）

## P1：候选位置来源

第一阶段只允许：

- 同图左右相邻位置；
- 少量同图其他相似区域；
- 不做跨图。

## P2：几何硬规则

候选位置必须同时满足：

- target_bbox 完整在图内；
- target_patch 完整在图内；
- 与任意 GT bbox IoU = 0；
- 不在图像边缘 3% 区域；
- 与最近 GT 的边缘距离：`>= max(4px, 0.15 * target_short)`；
- 单图最多新增 1 个目标（mini 阶段）；
- full 阶段默认 1 个，必要时最多 2 个，但 class 7 仍最多 1 个。

## P3：same-y-band

- tricycle：0.15
- awning：0.10
- bus：0.20

公式：

```text
abs(new_cy - src_cy) <= max(2px, same_y_ratio * src_h)
```

## P4：位置语义评分

```text
S_pos = 0.25 * S_empty
      + 0.20 * S_same_band
      + 0.20 * S_context
      + 0.15 * S_scale
      + 0.10 * S_texture
      + 0.10 * S_density
```

### 初始阈值

- tricycle：`S_pos >= 0.68`
- awning：`S_pos >= 0.78`
- bus：`S_pos >= 0.62`

### 解释

- awning 最脆弱，因此位置阈值最高；
- bus 最稳，因此阈值最低；
- tricycle 居中。

---

## 7. 生成规则（R 组）

## R1：生成框 target_bbox 先定，再生成

所有 ReCon 生成都不是“给图随便出目标”，而是：

- 先选位置；
- 再确定 `target_bbox`；
- 然后在这个框内生成目标。

## R2：target_bbox 由尺寸模板控制

### 初始尺寸模板

| 类别 | target_short 参考区间 | target_aspect_ratio |
|---|---:|---:|
| tricycle | 16–40 | 0.5–2.8 |
| awning-tricycle | 20–44 | 0.6–2.6 |
| bus | 24–120 | 1.5–5.5 |

### 规则

- 继承源实例的宽高比；
- 只允许 0.85–1.15 的小范围缩放；
- 不允许极端拉伸或压缩。

## R3：ReCon 模式

### Mode A：Boundary Repair

适用：
- tricycle
- awning-tricycle
- small bus

参数：
- `denoise_strength = 0.15–0.25`
- 每位置生成 4 个版本

### Mode B：Object + Boundary Refinement

适用：
- bus
- 高质量 tricycle

参数：
- `denoise_strength = 0.25–0.35`
- 每位置生成 4–8 个版本

### Mode C：Pure Region Generation

第一阶段只做对照，不作为主训练集来源。

## R4：结构化 prompt

### tricycle
positive:
`a small tricycle in aerial view, realistic UAV image, on urban road, natural shadow`

negative:
`motorcycle, bicycle, bus, car, blurred, deformed, extra vehicle, artifact`

### awning-tricycle
positive:
`a small covered tricycle with canopy in aerial view, realistic UAV image, on urban road, clear canopy structure`

negative:
`ordinary tricycle, motorcycle, bicycle, missing canopy, deformed vehicle, extra vehicle, artifact`

### bus
positive:
`a bus in aerial view, realistic UAV image, on road, rectangular body, natural shadow`

negative:
`truck, van, car, deformed bus, extra vehicle, artifact`

---

## 8. 质量治理规则（Q 组）

## Q1：Grounding DINO 验收

| 类别 | IoU(target_bbox, det_bbox) | score |
|---|---:|---:|
| tricycle | >=0.45 | >=0.20 |
| awning-tricycle | >=0.40 | >=0.18 |
| bus | >=0.50 | >=0.25 |

如果 patch 中新增 vehicle 数量 > 1 且不属于已有 GT，则拒绝。

## Q2：SAM 验收

| 类别 | IoU(SAM_bbox, target_bbox) | mask_area/bbox_area |
|---|---:|---:|
| tricycle | >=0.60 | 0.25–0.90 |
| awning-tricycle | >=0.65 | 0.30–0.85 |
| bus | >=0.60 | 0.35–0.95 |

允许用 SAM 更新 bbox，但必须满足：

- `IoU(new_bbox, target_bbox) >= 0.70`
- `area_ratio in [0.75, 1.35]`
- `center_shift <= 0.20 * bbox_diag`

## Q3：YOLO26 baseline 验收

- A：YOLO26 正确检出 + GDINO/SAM 通过 → 少量保留
- B：YOLO26 检不出，但 GDINO/SAM 通过 → 高价值难例，优先保留
- C：YOLO26 与 GDINO 都检不出 → 丢弃
- D：YOLO26 错类 → 6/7 直接丢弃；8 类可人工抽查
- E：生成后引入额外高置信误检 → 丢弃

## Q4：最终总分

```text
Q_final = 0.20 * Q_src
        + 0.20 * S_pos
        + 0.20 * Q_gdino
        + 0.20 * Q_sam
        + 0.15 * Q_yolo
        + 0.05 * Q_diversity
```

### 初始阈值

- tricycle：`Q_final >= 0.74`
- awning-tricycle：`Q_final >= 0.82`
- bus：`Q_final >= 0.70`

### 说明

- awning 最严格，因为它最容易被错误增强伤到；
- bus 最低，因为它更稳；
- 最终阈值仍要通过 mini 阶段的阈值敏感性验证锁死。

---

## 9. 类别平衡核心规则（C 组）

## C1：新增比例不能写死，要先统计再反推

C1 不能直接写成：
- 6:+10~20%
- 7:+5~10%
- 8:+20~30%

因为只看真实 train 数量时，真正的缺口顺序是：

- 7 最缺
- 6 次之
- 8 最不缺

所以 C1 应该由四类量共同决定：

- `N_c`：类实例总数
- `I_c`：正样本图像数
- `Q_c`：可用源池大小
- `p_c`：生成通过率

### Stage-1 初始建议

| 类别 | 建议新增比例 |
|---|---:|
| tricycle (6) | +8% ~ +12% |
| awning-tricycle (7) | +6% ~ +10% |
| bus (8) | +5% ~ +8% |

### Stage-2（仅在 Stage-1 有效后）累计上限

| 类别 | 累计上限 |
|---|---:|
| tricycle (6) | +15% ~ +20% |
| awning-tricycle (7) | +15% ~ +25% |
| bus (8) | +8% ~ +12% |

## C2：每图新增目标数不能永远固定 1 个

### mini 阶段
- 每图严格 1 个

### full 阶段
- 默认 1 个
- 满足高质量位置条件时，6/8 可到 2 个
- 7 仍最多 1 个

## C3：每个源实例最多使用几次，不该一刀切

### 最终建议

| 类别 | max_use |
|---|---:|
| awning-tricycle (7) | 2 |
| tricycle (6) | 3 |
| bus (8) | 4 |

但当 `max_use > 2` 时，必须强制满足同源多样化约束：

- 位置上下文不同；
- target_bbox 尺寸差异 >= 10%；
- prompt 变体不同；
- seed 不同；
- ReCon 模式不同；
- 视觉特征相似度不过高。

例如可用：
- `LPIPS >= 0.12` 或
- `DINO cosine distance >= 0.08`

---

## 10. mini → full 两阶段实施

## 10.1 mini 阶段

目标：
- 验证流程是否真的能提升，而不是一开始追求数量；
- 先把 C1 / C2 / C3 / Q_final 阈值跑出来。

建议：

- 三类都生成；
- 每类少量；
- 每图只加 1 个；
- 每位置生成 4–8 个版本；
- 通过 QA 后形成 mini 增强集；
- 先跑短程训练或快速验证。

## 10.2 full 阶段

只有当 mini 有效后才进入 full：

- 全量扫描原始数据；
- 全量构建实例库；
- 全量候选位置生成；
- 全量 ReCon 出图；
- 仍只保留 pass 图像与标签；
- 用 `train_union_full.txt` 训练。

---

## 11. 数据组织与训练方式

### 最终目录建议

```text
/home/shukang/datasets/VisDrone/                      # 原始集，不动
/home/shukang/datasets/VisDrone_ReCon_Pass/          # 只存 pass 图像和标签
/home/shukang/datasets/split_files/train_raw.txt
/home/shukang/datasets/split_files/train_recon_pass.txt
/home/shukang/datasets/split_files/train_union.txt
/home/shukang/datasets/split_files/val_raw.txt
```

### 训练方式

- 原始图像不复制；
- 只保存 pass 图像和标签；
- `train_union.txt = train_raw.txt + train_recon_pass.txt`；
- YAML 直接指向 union txt。

---

## 12. 推荐实验顺序

### Exp-0
raw baseline

### Exp-1
ReCon-QA-678-mini

### Exp-2
ReCon-QA-678-balanced-v1

### Exp-3
ReCon-QA-678-balanced-v2

### Exp-4
Strict awning ablation

重点不是一开始做最全，而是：

> 先把“样本做对”，再把“数量做够”。

---

## 13. 最终一句话版

> 本方案的最终核心，不是“多加几个尾类目标”，而是“围绕真实高质量 6/7/8 实例，先用 SAM 保证拿到纯目标，再用 Grounding DINO 找到语义合理位置，再由 ReCon 在给定生成框内做局部受控生成，最后用 Grounding DINO、SAM、YOLO26 三重机制严格筛掉坏样本，只把真正高质量、可学习、位置正确的新样本并入训练集”。
