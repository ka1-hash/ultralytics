# Bus vs Truck ReCon 处理核心脚本目录对比分析

---

## 一、目录总览

### 1.1 数据集配置目录 (`split_files_*`)

| 目录 | 用途 | 训练集行数 | 增强方式 |
|------|------|-----------|---------|
| `split_files_busonly/` | BusOnly 基线（仅原始数据） | 6,870 | 无 |
| `split_files_busfirst_v47/` | BusFirst V47（InPlace 增强） | 6,747 | InPlace |
| `split_files_busfirst_v47_chip/` | BusFirst Chip（InPlace + Chip） | 6,789 | InPlace + Chip |
| `split_files_truck/` | Truck（InPlace + Chip） | 7,794 | InPlace + Chip |
| `split_files_truck_full/` | Truck Full（仅 InPlace，无 Chip） | 较多 | InPlace |

### 1.2 工具脚本目录 (`toolkit/`)

| 目录 | 用途 | 配置 |
|------|------|------|
| `toolkit/recon_busfirst/` | Bus ReCon 初版 | 多配置文件 |
| `toolkit/recon_bus_harmonize_busfirst_v47/` | Bus V47 最终版（⏳ 当前训练中） | `config_busfirst.yaml` |
| `toolkit/recon_truck/` | Truck ReCon（✅ 已完成） | `config_truck.yaml` |
| `toolkit/recon_inplace_v5/` | InPlace V5 通用工具 | `config_inplace_v5.yaml` |

---

## 二、详细目录结构

### 2.1 BusFirst V47 Chip 目录 (`split_files_busfirst_v47_chip/`)

```
split_files_busfirst_v47_chip/
├── VisDrone_InPlaceChip_cls8.yaml          # 训练数据配置
├── train_union_inplace_chip_cls8.txt        # 联合训练集：原始 + InPlace + Chip
├── train_aug_only_withchips_cls8.txt        # 仅增强数据：InPlace + Chip
```

**数据量统计：**
- 原始数据：6,870 张
- InPlace 增强：276 张 (class 8: bus)
- Chip 切片增强：42 张 (class 8: bus)
- **总训练集：6,789 张**（原始 + 增强合并去重）

**YAML 配置：**
```yaml
path: /home/shukang/project/recon_workbench/datasets/busfirst_v47_chip
train: train_union_inplace_chip_cls8.txt
val: /home/shukang/project/recon_workbench/datasets_in/VisDrone/images/val
names: 10类 (0-9)
```

### 2.2 Truck 目录 (`split_files_truck/`)

```
split_files_truck/
├── VisDrone_InPlaceChip_cls5.yaml           # 训练数据配置
├── train_union_inplace_chip_cls5.txt         # 联合训练集：原始 + InPlace + Chip
├── train_aug_only_withchips_cls5.txt         # 仅增强数据：InPlace + Chip
```

**数据量统计：**
- 原始数据：6,870 张
- InPlace 增强：1,037 张 (class 5: truck)
- Chip 切片增强：286 张 (class 5: truck)
- **总训练集：7,794 张**（原始 + 增强合并去重）

**YAML 配置：**
```yaml
path: /home/shukang/project/recon_workbench/datasets/v5_truck
train: train_union_inplace_chip_cls5.txt
val: /home/shukang/project/recon_workbench/datasets_in/VisDrone/images/val
names: 10类 (0-9)
```

### 2.3 BusFirst V47 目录 (`split_files_busfirst_v47/`)

```
split_files_busfirst_v47/
├── VisDrone_AllTail.yaml
├── train_union_all_tail.txt
├── val_raw_all_tail.txt
```

**数据量：** 6,747 张（原始 + InPlace，无 Chip）

### 2.4 BusOnly 目录 (`split_files_busonly/`)

```
split_files_busonly/
├── VisDrone_AllTail.yaml
├── train_union_all_tail.txt
├── val_raw_all_tail.txt
```

**数据量：** 6,870 张（仅原始数据，无增强）

---

## 三、核心脚本目录对比

### 3.1 Bus ReCon 工具链 (`toolkit/recon_bus_harmonize_busfirst_v47/`)

```
recon_bus_harmonize_busfirst_v47/
├── auto_v47.sh              # 自动化执行脚本
├── build_all_tail.py        # 构建训练集列表
├── build_busfirst.py        # BusFirst 入口
├── build_insitu.py          # InPlace 构建
├── build_v47.sh             # V47 构建脚本
├── check_mainline.py        # 主线检查
├── common.py                # 通用工具函数
├── config_busfirst.yaml     # ⭐ 核心配置
├── merge_bus_only.py        # 合并工具
├── model_loader.py          # 模型加载器
├── recon_engine.py          # ⭐ ReCon 引擎
└── train_v47.sh             # 训练脚本
```

### 3.2 Truck ReCon 工具链 (`toolkit/recon_truck/`)

```
recon_truck/
├── build_all_tail.py        # 构建训练集列表
├── build_busfirst.py        # 构建入口（复用BusFirst框架）
├── common.py                # 通用工具函数
├── config_truck.yaml        # ⭐ 核心配置（含chip_post）
├── merge_bus_only.py        # 合并工具
├── model_loader.py          # 模型加载器
├── recon_engine.py          # ⭐ ReCon 引擎
├── verify2.py               # 验证脚本
└── verify_truck.py          # Truck 验证脚本
```

### 3.3 InPlace V5 通用工具 (`toolkit/recon_inplace_v5/`)

```
recon_inplace_v5/
├── build_inplace_v5.py      # InPlace 构建
├── common.py                # 通用工具
├── config_inplace_v5.yaml   # InPlace 配置
├── fast_verify_v5.py        # 快速验证
├── model_loader.py          # 模型加载器
├── post_inplace_chip_v5.py  # ⭐ Chip 切片增强核心逻辑
├── recon_engine.py          # ReCon 引擎
├── run_full_inplace_plus_chip_v5.sh  # 全流程执行
├── run_full_inplace_v5.sh   # 仅 InPlace
├── run_post_chip_inplace_v5.sh       # 仅 Chip
└── run_verify_inplace_v5.sh # 验证
```

---

## 四、配置参数对比

### 4.1 ReCon 核心参数

| 参数 | Bus (config_busfirst.yaml) | Truck (config_truck.yaml) |
|------|---------------------------|--------------------------|
| **目标类别** | `[8]` (bus) | `[5]` (truck) |
| **recon_strength** | 0.80 (范围 0.72-0.86) | 0.80 (固定) |
| **object_core_fuse** | 0.65 | 0.65 |
| **boundary_band_fuse** | 0.55 | 0.55 |
| **num_versions** | 2 | 2 |
| **exec_patch_size** | 512 | 512 |
| **guidance_scale** | 4.0 | 4.0 |
| **num_inference_steps** | 24 | 24 |
| **prompt** | "a clear bus in UAV view, preserve scene geometry" | "a clear truck in UAV view, preserve scene geometry" |
| **max_sources** | 1,600 (class 8) | 未明确限制 |
| **source_quality_lambda** | 0.15 | 无 |
| **patch_area_ratio** | 0.20 (class 8) | 无 |
| **patch_min_size** | 160 (class 8) | 无 |
| **patch_max_size** | 320 (class 8) | 无 |

### 4.2 Chip 切片增强参数

| 参数 | Bus | Truck |
|------|-----|-------|
| **chip_post.enabled** | ✅ | ✅ |
| **chip_post.class_ids** | `[8]` (bus) | `[5, 6]` (truck + tricycle) |
| **chip_post.imgsz** | 1280 | 1280 |
| **chip_post.ref_long** | 1920 | 1920 |
| **chip_post.small_thr** | 16.0 | 16.0 |
| **chip_post.allow_swapped_shape** | true | true |

---

## 五、关键差异分析

### 5.1 数据量差异

| 指标 | Bus | Truck | 差异 |
|------|-----|-------|------|
| 原始数据 | 6,870 | 6,870 | 相同 |
| InPlace 增强 | 276 | 1,037 | Truck 多 3.8× |
| Chip 切片 | 42 | 286 | Truck 多 6.8× |
| 总训练集 | 6,789 | 7,794 | Truck 多 1,005 |

### 5.2 增强策略差异

1. **ReCon 强度**：两者 `recon_strength` 相同（0.80），但 Bus 在 `config_busfirst.yaml` 中设置了更精细的 patch 控制（`patch_area_ratio`, `patch_min/max_size`, `occupancy`），而 Truck 的 `config_truck.yaml` 没有这些参数。

2. **Chip 目标类别**：Truck 对 `[5, 6]` (truck + tricycle) 都做切片，Bus 仅对 `[8]` (bus) 做切片。

3. **增强数量**：Truck 的增强数据是 Bus 的 3.5 倍（1,323 vs 318），可能因：
   - Truck 的 source 筛选条件更宽松
   - 没有 `max_sources` 限制
   - 缺少 `patch_area_ratio` 等精细控制

4. **质量控制**：Bus 的 `config_busfirst.yaml` 包含完整的 QC 体系（`qc` 段），Truck 的 `config_truck.yaml` 缺少这些参数。

### 5.3 训练结果对比

| 实验 | 训练集 | mAP50 | mAP50-95 | 耗时 |
|------|--------|-------|----------|------|
| BusOnly | 6,870 | 0.622 | - | 120 epochs |
| BusFirst V47 | 6,747 | **0.619** | - | 120 epochs |
| BusFirst Chip | 6,789 | 0.445 | 0.268 | 训练中 |
| **Truck** | 7,794 | **0.468** | 0.284 | 102 epochs |

### 5.4 脚本结构差异

| 功能 | Bus | Truck |
|------|-----|-------|
| 自动化脚本 | ✅ `auto_v47.sh`, `build_v47.sh`, `train_v47.sh` | ❌ 无 |
| 主线检查 | ✅ `check_mainline.py` | ❌ 无 |
| InPlace 构建 | ✅ `build_insitu.py` | ❌ 复用 `recon_inplace_v5/` |
| 验证脚本 | ❌ | ✅ `verify2.py`, `verify_truck.py` |
| 配置文件 | 1 个（`config_busfirst.yaml`） | 1 个（`config_truck.yaml`） |

---

## 六、训练脚本对比

### Bus Chip 训练脚本
**文件：** `/home/shukang/project/YOLO26/train_recon_busfirst_chip_cls8.sh`
```bash
DATA="/home/shukang/project/recon_workbench/split_files_busfirst_v47_chip/VisDrone_InPlaceChip_cls8.yaml"
MODEL="yolo26n-nop5"
BATCH=4, DEVICE=0, EPOCHS=120, PATIENCE=10, IMGSZ=1280, MULTI_SCALE=0
```

### Truck 训练脚本
**文件：** `/home/shukang/project/YOLO26/train_recon_truck_cls5.sh`
```bash
DATA="/home/shukang/project/recon_workbench/split_files_truck/VisDrone_InPlaceChip_cls5.yaml"
MODEL="yolo26n-nop5"
BATCH=4, DEVICE=0, EPOCHS=120, PATIENCE=10, IMGSZ=1280, MULTI_SCALE=0
```

> **训练参数完全一致**，仅数据路径不同。

---

## 七、总结

| 维度 | Bus | Truck |
|------|-----|-------|
| 配置复杂度 | 高（精细 QC + patch 控制） | 低（基础配置） |
| 增强数据量 | 适中（318 张） | 较多（1,323 张） |
| Chip 切片 | 少（42 张） | 多（286 张） |
| 自动化程度 | 高（全套 shell 脚本） | 低（依赖手动） |
| 最终 mAP50 | 0.619 (V47) | 0.468 |
| 训练状态 | ⏳ 进行中（Chip） | ✅ 已完成 |