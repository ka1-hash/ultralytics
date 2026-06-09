# YOLO26 项目完整分析报告

> 生成日期：2026-06-09  
> 分析范围：`/home/shukang/project/YOLO26` + 关联的 `recon_workbench` 数据管线

---

## 一、项目整体架构

```
YOLO26/
├── train.py                    # 主训练入口（旧版）
├── train_tail5.py              # 增强训练入口（新版：+--lr0, --weights_init, --freeze, --optimizer）
├── train-old.py                # 更早版本的训练脚本
├── train_busfirst.py           # BusFirst 专用训练脚本
│
├── ultralytics/                # ultralytics 8.4.37 定制版（含 YOLO26 模型定义）
│   ├── cfg/models/26/          # yolo26n-nop5.yaml 等模型配置
│   ├── cfg/datasets/           # 标准数据集 yaml（VisDrone.yaml 等）
│   ├── engine/                 # trainer, validator, predictor
│   └── models/yolo/            # YOLO 检测模型实现
│
├── ReCon_A100_Standalone/      # ReCon A100 独立部署包
│   ├── recipes/                # 实验配方：
│   │   ├── YOLO26_ReCon_Balance678_3pct/   # 尾类均衡（tricycle/awning-tricycle/bus 各增到 3%）
│   │   └── YOLO26_ReCon_SI_LAG_minimal/    # SI-LAG 最小充分版
│   └── venv/                   # A100 上自带的虚拟环境
│
├── log/                        # 新版训练日志（train_tail5.py，A100 为主，含完整验证 mAP）
├── logs/                       # 旧版训练日志（train.py）
├── sh/ / sh_old/               # 辅助脚本（bash 封装）
├── weights/                    # 预训练权重
├── datasets/                   # 数据集配置文件（软链接/本地配置）
└── runs/                       # 训练输出（已清空）
```

---

## 二、核心训练脚本

### 2.1 train.py（旧版入口）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolo26n-nop5` | 模型名 |
| `--data` | `VisDrone_ReCon` | 数据集名 |
| `--batch` | `4` | 批次大小 |
| `--device` | `0,1` | GPU 设备 |
| `--imgsz` | `1280` | 输入尺寸 |
| `--epochs` | `120` | 训练轮数 |
| `--patience` | `10` | Early stopping 耐心值 |
| `--project` / `--name` | 自动生成 | 输出目录 |

### 2.2 train_tail5.py（新版入口）

在 `train.py` 基础上新增：

| 参数 | 说明 |
|------|------|
| `--lr0` | 初始学习率（默认 `0.01`） |
| `--weights_init` | 从已有 best.pt / last.pt 初始化权重 |
| `--freeze` | 冻结前 N 层 |
| `--optimizer` | 优化器选择（默认 `MuSGD`） |
| `--name_prefix` | 自定义名称前缀 |

---

## 三、通用训练超参数

所有实验统一使用以下配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | `yolo26n-nop5.yaml` | YOLO26 nano，无 P5 检测头 |
| 预训练权重 | `weights/yolo26n.pt` | 从头加载（408/568 层迁移） |
| 输入尺寸 | `1280` | 正方形输入 |
| 批次大小 | `4` | 每 GPU |
| 训练轮数 | `120` | epochs |
| Early Stopping | `patience=10` | 10 epoch 无提升则停止 |
| 优化器 | `MuSGD` | 动量 SGD，`momentum=0.937` |
| 学习率 | `lr0=0.01, lrf=0.01` | 余弦退火 |
| 数据增强 | `mosaic=1.0, close_mosaic=10, erasing=0.4, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` | |
| 缓存 | `cache=disk` | 磁盘缓存以加速训练 |

---

## 四、全部实验一览

### 4.1 log/ 目录（新版：train_tail5.py 驱动）

| # | 实验名 | 日期 | GPU | 训练脚本 | YAML 配置 | 训练集位置 | 状态 |
|---|--------|------|-----|----------|-----------|-----------|------|
| 1 | **Tail5_cls6（初版）** | 2026-06-07 | A100 | train_tail5.py | `split_files_tail5/VisDrone_Tail5Chip_cls6.yaml` | VisDrone 原图 + Tail5 InPlace Chips，6 cls | ❌ 参数错误 |
| 2 | **Tail5_cls6（重试）** | 2026-06-07 | A100 | train_tail5.py | 同上 | 同上 | ✅ |
| 3 | **AllChipFinal_cls6** | 2026-06-07 | A100 | train_tail5.py | `split_files_tail5/VisDrone_AllChipFinal_cls6.yaml` | VisDrone 原图 + 全部芯片，6 cls | ✅ |
| 4 | **AllChipFinal_cls7** | 2026-06-08 | A100 | train_tail5.py | `split_files_tail5/VisDrone_AllChipFinal_cls7.yaml` | VisDrone 原图 + 全部芯片，7 cls | ✅ |
| 5 | **ABonly** | 2026-06-08 | A100 | train_tail5.py | `split_files_tail5/VisDrone_ABonly.yaml` | VisDrone 原图全量 + raw_chips | ✅ |
| 6 | **BaselineRetrain** | 2026-06-09 | A100 | train_tail5.py | `split_files_tail5/VisDrone_BaselineRetrain.yaml` | 纯 VisDrone 原图重训 | ✅ |
| 7 | **SOChipRank_keep010** | 2026-05-14 | RTX 2080 Ti ×2 | train.py | `VisDrone_SOChipRank_1280_keep010.yaml` | VisDrone 10% 切片子集 [已删除] | ✅ |

### 4.2 logs/ 目录（旧版：train.py 驱动）

| # | 实验名 | 日期 | GPU | 训练脚本 | YAML 配置 | 训练集位置 | 状态 |
|---|--------|------|-----|----------|-----------|-----------|------|
| 8 | **baseline** | 2026-04-16 | RTX 2080 Ti ×2 | train.py | `ultralytics/cfg/datasets/VisDrone.yaml` | VisDrone 原始数据集 | ✅ |
| 9 | **AllTail** | 2026-05-31 | A100 | train.py | `split_files/VisDrone_AllTail.yaml` | VisDrone + 所有尾部芯片工会 | ✅ |
| 10 | **BusOnly** | 2026-06-01 | A100 | train.py | `split_files_busonly/VisDrone_AllTail.yaml` | VisDrone + Bus 芯片工会 | ✅ |
| 11 | **BusFirst_v47** | 2026-06-02 | A100 | train.py | `split_files_busfirst_v47/VisDrone_AllTail.yaml` | BusFirst v47 数据集 | ✅ |
| 12 | **TruckFirst** | 2026-06-04 | A100 | train.py | `split_files_truck/VisDrone_InPlaceChip_cls5.yaml` | TruckFirst 数据集，5 cls | ✅ |
| 13 | **BusFirstChip** | 2026-06-05 | A100 | train.py | `split_files_busfirst_v47_chip/VisDrone_InPlaceChip_cls8.yaml` | BusFirst Chip 数据集，8 cls | ✅ |

### 4.3 Recipe 配方实验（位于 ReCon_A100_Standalone）

| # | 配方 | 数据集 | 数据路径 | 训练脚本 |
|---|------|--------|----------|----------|
| 14 | **ReCon_Balance678_3pct** | `VisDrone_ReCon_Balance678_3pct_1280.yaml` | `/home/shukang/datasets/VisDrone_ReCon_Balance678_3pct_1280/` | `recipes/YOLO26_ReCon_Balance678_3pct/scripts/recon_balance678/train_recon_balance678_square_1280.sh` |
| 15 | **ReCon_SI_LAG_minimal** | `VisDrone_ReCon_SI_LAG_minimal_1280.yaml` | `/home/shukang/datasets/VisDrone_ReCon_SI_LAG_minimal_1280/` | `recipes/YOLO26_ReCon_SI_LAG_minimal/recon_si_lag_minimal_package/scripts/recon_si_lag_minimal/train_recon_si_lag_square_1280.sh` |

---

## 五、各实验数据集构成详解

### 5.1 baseline（实验 #8）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `ultralytics/cfg/datasets/VisDrone.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/datasets_in/VisDrone` |
| 训练集 | `images/train/` — VisDrone 全量原图 |
| 验证集 | `images/val/` — VisDrone 验证集 |
| 类别 | 10 |
| 训练图片数 | VisDrone 训练集全量 |
| 日志文件 | `YOLO26/logs/baseline.log` |

### 5.2 SOChipRank_keep010（实验 #7）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `VisDrone_SOChipRank_1280_keep010.yaml` |
| 数据路径 | `/home/shukang/datasets/VisDrone_SOChipRank_1280/datasets/VisDrone_SOChipRank_1280_keep010/`（已删除） |
| 训练集 | VisDrone 训练集中质量排名前 10% 的切片 |
| 生成方式 | `post_raw_chip_tail5.py --keep-ratio 0.10` |
| 日志文件 | `YOLO26/log/VisDrone_SOChipRank_1280_keep010-yolo26n-nop5-b4-s1280-ms0-20260514-214102.log` |
| 状态 | ⚠️ 数据已删除，日志保留 |

### 5.3 AllTail（实验 #9）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files/VisDrone_AllTail.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench` |
| 训练列表 | `split_files/train_union_all_tail.txt` |
| 构成 | VisDrone 原图 + 所有尾部类（tricycle, awning-tricycle, bus）的芯片并集 |
| 日志文件 | `YOLO26/logs/VisDrone_AllTail-yolo26n-nop5-b4-s1280-ms0-20260531-234534.log` |

### 5.4 BusOnly / BusFirst_v47（实验 #10, #11）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_busonly/VisDrone_AllTail.yaml` 或 `split_files_busfirst_v47/VisDrone_AllTail.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench` |
| 训练列表 | `train_union_all_tail.txt`（各版本不同） |
| 构成 | VisDrone 原图 + Bus 类芯片并集（BusFirst 版本比 BusOnly 新增更多 bus chips） |
| 日志文件 | `logs/VisDrone_BusOnly-yolo26n-nop5-b4-s1280-ms0-20260601-130524.log` / `logs/VisDrone_BusFirst_v47-yolo26n-nop5-b4-s1280-20260602-081822.log` |

### 5.5 TruckFirst（实验 #12）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_truck/VisDrone_InPlaceChip_cls5.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/datasets/v5_truck` |
| 训练列表 | `train_union_inplace_chip_cls5.txt` |
| 构成 | VisDrone 原图 + Truck 类 InPlace 芯片 |
| 类别 | 5（仅训练相关类别子集） |
| 日志文件 | `logs/VisDrone_TruckFirst-yolo26n-nop5-b4-s1280-ms0-20260604-221654.log` |

### 5.6 BusFirstChip（实验 #13）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_busfirst_v47_chip/VisDrone_InPlaceChip_cls8.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/datasets/busfirst_v47_chip` |
| 训练列表 | `train_union_inplace_chip_cls8.txt` |
| 构成 | VisDrone 原图 + BusFirst v47 InPlace 芯片 |
| 类别 | 8 |
| 日志文件 | `logs/VisDrone_BusFirstChip-yolo26n-nop5-b4-s1280-ms0-20260605-122130.log` |

### 5.7 Tail5Chip_cls6（实验 #1, #2）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_tail5/VisDrone_Tail5Chip_cls6.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/outputs/VisDrone_Tail5_InPlace_ReCon` |
| 训练列表 | `split_files_tail5/train_union_inplace_chip_cls6.txt` |
| 构成 | VisDrone 原图 + Tail5 InPlace Chips（仅 tricycle, awning-tricycle, bus 的切片） |
| 日志文件 | `log/VisDrone_Tail5_cls6-yolo26n-nop5-b4-s1280-ms0-20260607-160144.log` / `log/VisDrone_Tail5_cls6-yolo26n-nop5-b4-s1280-ms0-20260607-172645.log` |

### 5.8 AllChipFinal_cls6（实验 #3）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_tail5/VisDrone_AllChipFinal_cls6.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/outputs/VisDrone_Tail5_InPlace_ReCon` |
| 训练列表 | `split_files_tail5/train_final_allchip_cls6.txt` |
| 构成 | VisDrone 原图 + 全部 raw_chips + aug_chips（所有类别芯片，但只训练 6 类子集） |
| 日志文件 | `log/VisDrone_AllChip_cls6-yolo26n-nop5-b4-s1280-ms0-20260607-230815.log` |

### 5.9 AllChipFinal_cls7（实验 #4）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_tail5/VisDrone_AllChipFinal_cls7.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/outputs/VisDrone_Tail5_InPlace_ReCon` |
| 训练列表 | `split_files_tail5/train_final_allchip_cls7.txt` |
| 构成 | VisDrone 原图 + 全部 raw_chips + aug_chips（所有类别芯片，7 类子集） |
| 日志文件 | `log/VisDrone_AllChip_cls7-yolo26n-nop5-b4-s1280-ms0-20260608-115556.log` |

### 5.10 ABonly（实验 #5）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_tail5/VisDrone_ABonly.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/outputs/VisDrone_Tail5_InPlace_ReCon` |
| 训练列表 | `split_files_tail5/train_ABonly_allchip.txt` |
| 构成 | VisDrone 原图 6471 张 + raw_chips 567 张 = **7038 张** |
| 类别 | 10（全部类别） |
| 原图来源 | `datasets_in/VisDrone/images/train/` |
| 切片来源 | `outputs/VisDrone_Tail5_InPlace_ReCon/raw_chips/all/images/train/` |
| 日志文件 | `log/VisDrone_ABonly-yolo26n-nop5-b4-s1280-ms0-20260608-184641.log` |

### 5.11 BaselineRetrain（实验 #6）

| 项目 | 详情 |
|------|------|
| YAML 配置 | `split_files_tail5/VisDrone_BaselineRetrain.yaml` |
| YAML `path` | `/home/shukang/project/recon_workbench/datasets_in/VisDrone` |
| 训练列表 | `split_files_tail5/train_baseline_retrain.txt` |
| 构成 | 纯 VisDrone 原图（不含任何芯片/增强），作为 ABonly 的消融对照 |
| 日志文件 | `log/VisDrone_BaselineRetrain-yolo26n-nop5-b4-s1280-ms0-20260609-092807.log` |

---

## 六、验证 mAP50 演进记录

按 epoch 展示各实验的 mAP50（Val set: 548 images, 38759 instances, all classes mean）：

| Epoch | baseline | keep010 | TruckFirst | BusFirst_v47 | BusFirstChip | AllChip_cls7 | AllChip_cls6 | ABonly | BaselineRetrain |
|-------|----------|---------|------------|-------------|-------------|-------------|-------------|--------|-----------------|
| 1 | 0.115 | 0.116 | 0.135 | 0.126 | 0.126 | 0.132 | 0.159 | 0.131 | 0.123 |
| 5 | 0.172 | 0.183 | 0.212 | 0.198 | 0.198 | 0.214 | 0.237 | 0.207 | 0.184 |
| 10 | 0.214 | 0.229 | 0.256 | 0.236 | 0.236 | 0.253 | 0.271 | 0.248 | 0.237 |
| 20 | — | — | 0.281 | 0.271 | 0.271 | 0.296 | 0.318 | 0.285 | 0.271 |
| 30 | — | — | 0.305 | 0.297 | 0.307 | 0.318 | 0.337 | 0.302 | 0.294 |
| 40 | — | — | 0.324 | 0.314 | 0.344 | 0.341 | 0.364 | 0.327 | 0.309 |
| 50 | — | — | 0.342 | 0.350 | 0.350 | 0.361 | 0.384 | 0.340 | 0.323 |
| 60 | — | — | 0.355 | — | — | 0.378 | 0.392 | 0.350 | 0.342 |
| 70 | — | — | 0.364 | — | — | 0.361 | 0.373 | 0.364 | 0.353 |
| 80 | — | — | 0.369 | — | — | 0.360 | 0.384 | 0.374 | 0.364 |
| 90 | — | — | — | — | — | 0.378 | 0.400 | 0.395 | — |
| 100 | — | — | — | — | — | — | — | 0.400 | — |
| 110 | — | — | — | — | — | — | — | 0.408 | — |
| 120 | — | — | — | — | — | — | — | 0.414 | — |

> 注："—" 表示日志不包含该 epoch 数据或实验提前停止。

---

## 七、数据集关系图

```
原始 VisDrone (datasets_in/VisDrone/images/)
  │
  ├─► Train 全量（6471 张）──► 直接用于 baseline, BaselineRetrain, ABonly 等
  │
  ├─► Train 选取子集 ──► split_files/train_union_all_tail.txt ──► AllTail, BusOnly, BusFirst_v47
  │
  ├─► post_raw_chip_tail5.py ──► raw_chips/all/                   ← 对 VisDrone 图片切片
  │     │                          (outputs/VisDrone_Tail5_InPlace_ReCon)
  │     │
  │     ├─► keep_ratio=0.10 ──► keep010 数据集（647 张）          ← 10% 质量最优切片 [已删除]
  │     │
  │     ├─► aug_chips/class{6,7}/ ──► Tail5Chip_cls{6,7}         ← 仅尾类芯片增强
  │     │
  │     └─► raw_chips/all/ ──► AllChipFinal_cls{6,7}, ABonly     ← 全部芯片保留
  │
  ├─► BusFirst/TruckFirst 处理 ──► split_files_busfirst_v47_chip / split_files_truck
  │
  └─► Co-Mix 增强管道 ──► aug_chips/class_by_index/ ──► ReCon Balance678 / SI-LAG
```

---

## 八、数据可用性状态总表

| 数据类型 | 路径 | 状态 |
|----------|------|------|
| VisDrone 训练集原图 | `datasets_in/VisDrone/images/train/` | ✅ 存在 |
| VisDrone 验证集 | `datasets_in/VisDrone/images/val/` | ❌ 不存在 |
| VisDrone 测试集 | `datasets_in/VisDrone/images/test/` | ✅ 存在 |
| raw_chips | `outputs/VisDrone_Tail5_InPlace_ReCon/raw_chips/all/images/train/` | ✅ 存在（含 .jpg + .npy） |
| aug_chips | `outputs/VisDrone_Tail5_InPlace_ReCon/aug_chips/class{6,7}/` | ✅ 存在 |
| keep010 数据集 | `/home/shukang/datasets/VisDrone_SOChipRank_1280/` | ❌ 已删除 |
| 所有训练权重 | `runs/detect/visdrone*/` | ❌ 已删除 |
| 训练日志 | `log/` + `logs/` | ✅ 完整保留 |
| YAML 配置 | `split_files_tail5/`, `split_files*/` | ✅ 完整保留 |
| 训练列表文件 | `train_*_allchip.txt` 等 | ✅ 完整保留 |
| ReCon Balance678 数据集 | `/home/shukang/datasets/VisDrone_ReCon_Balance678_3pct_1280/` | ❓ 未验证 |
| ReCon SI-LAG 数据集 | `/home/shukang/datasets/VisDrone_ReCon_SI_LAG_minimal_1280/` | ❓ 未验证 |

---

## 九、实验设计逻辑

### 9.1 实验演进脉络

```
Phase 1: 基线建立
  baseline (VisDrone bare) ──► 确立 YOLO26n 在 VisDrone 上的 baseline 性能

Phase 2: 10% 切片子集快速实验
  SOChipRank_keep010 ──► 验证切片质量排序 + 小样本训练可行性

Phase 3: 尾部类芯片补充
  AllTail ──► 所有尾部芯片并集，观察整体提升
  BusOnly ──► 仅 Bus 类芯片，观察单类增益
  BusFirst_v47 ──► 扩大 Bus 芯片数量（v47 版本）
  TruckFirst ──► Truck 类芯片，5 cls 子集

Phase 4: Tail5 芯片 + InPlace 策略
  BusFirstChip ──► Bus chip 8 cls，细化分类
  Tail5Chip_cls6 ──► tail5 类 InPlace chips，6 cls
  Tail5Chip_cls7 ──► tail5 类 InPlace chips，7 cls

Phase 5: 全集对照 + 消融
  AllChipFinal_cls6 ──► 所有芯片 + aug，6 cls
  AllChipFinal_cls7 ──► 所有芯片 + aug，7 cls
  ABonly ──► 所有芯片，10 cls（全集对照）
  BaselineRetrain ──► 纯原图重训（消融对照，验证芯片增益）
```

### 9.2 ABonly 实验定位

**AB** = **All Chips + All Classes**。这是 `Tail5Chip` 和 `AllChipFinal` 系列实验的**全集对照实验**：

- `Tail5Chip_cls{6,7}`：只保留 tail5 类的切片
- `AllChipFinal_cls{6,7}`：对所有类保留切片，但只训练对应子集的类别
- **ABonly**：对所有类保留切片，训练**全部 10 个类别**

ABonly = VisDrone 原图全部 + 所有 raw chips（10% keep ratio 切片），用于衡量 "在完整 10 类训练中加入所有芯片" 的综合效果。

---

## 十、关键发现与建议

1. **验证集缺失**：`datasets_in/VisDrone/images/val/` 目录不存在。如需重新训练或验证，必须先恢复 VisDrone 验证集。

2. **训练权重丢失**：所有 `runs/` 目录下的 best.pt / last.pt 已删除，仅保留训练日志。如需复现结果，需从头训练。

3. **ABonly 实验结果**：最终 mAP50=0.414（Epoch 120），相比 baseline 有提升。但 BaselineRetrain 仅训练到约 Epoch 80，需继续跑完才能做公平的消融对比。

4. **keep010 数据集已删除**：10% 切片子集（SOChipRank_1280_keep010）在 `/home/shukang/datasets/` 中不存在，但 raw_chips 源数据仍在 `outputs/` 中，可重新生成。

5. **ReCon 配方数据集**：Balance678 和 SI-LAG 数据集是否存在于 `/home/shukang/datasets/` 中待验证。若不存在，需用 ReCon 相关脚本重新构建。

6. **芯片增益趋势**：从 AllChip_cls6（mAP50≈0.400 at E50）vs ABonly（mAP50≈0.340 at E50）的数据看，**子类训练（6 cls）比全类训练（10 cls）在芯片增益上表现更好**，说明芯片对特定尾部类的帮助更显著。