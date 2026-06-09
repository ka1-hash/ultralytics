# ReCon 类别平衡脚本说明文档（终版）

## 1. 文档定位

本文档只解释脚本，不重复完整方案本体。目标是让你快速理解：

1. 每个 Python 脚本做什么；
2. 每个 shell 脚本做什么；
3. 数据集是怎么一步步构建出来的；
4. mini / full 阶段分别如何运行；
5. 最终训练如何接入 `raw + recon_pass` 联合清单。

脚本目录如下：

```text
recon_balance_scripts/
├── 00_config_example.yaml
├── common.py
├── 01_build_instance_bank.py
├── 02_build_position_candidates.py
├── 03_make_recon_tasks.py
├── 04_run_recon_batch.py
├── 05_qc_recon_outputs.py
├── 06_assemble_balance_dataset.py
├── 07_make_split_lists.py
└── sh/
    ├── 08_build_dataset_mini.sh
    ├── 09_build_dataset_full.sh
    ├── 10_write_data_yaml.sh
    ├── 11_train_mini.sh
    ├── 12_train_full.sh
    ├── 13_run_mini_pipeline.sh
    └── 14_run_full_pipeline.sh
```

---

## 2. Python 脚本说明

## 2.1 00_config_example.yaml

### 作用
统一配置文件，所有 Python 与 sh 脚本都从这里读路径、类别规则、阈值、stage 设置。

### 主要内容

- 原始数据集路径
- 工作目录路径
- 6/7/8 类名称映射
- S2 / S3 / S4 规则
- 位置规则
- ReCon 规则
- QA 规则
- mini / full 两阶段配置

### 你要改什么

优先改：
- `dataset_root`
- `project_root`
- `output_root`
- `class_names`
- `s2_short_train_ranges`
- `recon_rules`
- `stage_configs`

---

## 2.2 common.py

### 作用
提供公共函数，减少各脚本重复代码。

### 主要函数类型

- 读取 yaml / csv / jsonl
- 读图像尺寸
- 读 yolo label
- 计算 `short_train`
- 计算 IoU
- 保存 csv / jsonl
- 解析阶段 `mini/full`
- 常用路径拼接

### 你要理解什么

`common.py` 本身不直接产出数据，它是整套流程的公共工具层。

---

## 2.3 01_build_instance_bank.py

### 作用
构建源实例库。

### 它做什么

1. 扫描原始 train 图像与标签；
2. 只提取 6/7/8 类实例；
3. 计算基础几何量：bbox、short_train、边界距离、邻域目标数等；
4. 调用 SAM（当前脚本里是占位接口）得到分割质量指标；
5. 根据 S2 / S3 / S4 规则筛选高质量实例；
6. 输出 `source_instances.csv`。

### 输入

- 原始图像目录
- 原始标签目录
- 配置文件

### 输出

- `metadata/source_instances_raw.csv`
- `metadata/source_instances_pass.csv`

### 关键字段

- `class_id`
- `image_id`
- `x1,y1,x2,y2`
- `short_train`
- `mask_bbox_iou`
- `mask_area_ratio`
- `largest_cc_ratio`
- `cc_count`
- `Q_src`
- `pass_s2`
- `pass_s4`
- `pass_qsrc`

### 你要看什么

这个脚本产出的 `source_instances_pass.csv` 就是后面所有生成任务的源池。

---

## 2.4 02_build_position_candidates.py

### 作用
在原图中为每个源实例找候选生成位置。

### 它做什么

1. 读取 `source_instances_pass.csv`；
2. 继承旧 SI-OCP 的思路，在同图左右相邻区域扫描候选框；
3. 检查几何合法性：不重叠、不越界、same-y-band 等；
4. 计算位置评分 `S_pos`；
5. 为每个源实例保留 top-k 候选位置；
6. 输出 `candidate_positions.csv`。

### 输入

- `source_instances_pass.csv`
- 原始图像尺寸信息
- 原始 GT 标签

### 输出

- `metadata/candidate_positions.csv`

### 关键字段

- `source_instance_row_id`
- `source_class_id`
- `image_id`
- `target_x1,target_y1,target_x2,target_y2`
- `S_empty`
- `S_same_band`
- `S_context`
- `S_scale`
- `S_texture`
- `S_density`
- `S_pos`

### 你要看什么

这个脚本决定“车放在哪里”。如果这里做错，后面的 ReCon 就算生成得很真，也还是会学到错位置。

---

## 2.5 03_make_recon_tasks.py

### 作用
把候选位置整理成可执行的 ReCon 任务列表。

### 它做什么

1. 读取 `candidate_positions.csv`；
2. 按类别选择 patch size、ReCon 模式；
3. 生成 structured prompt；
4. 把每个候选写成任务；
5. 输出 `recon_tasks.jsonl`。

### 输入

- `candidate_positions.csv`
- 配置文件

### 输出

- `metadata/recon_tasks.jsonl`

### 关键字段

- `task_id`
- `source_image_id`
- `target_image_id`
- `class_id`
- `source_bbox`
- `target_bbox`
- `patch_size`
- `recon_mode`
- `positive_prompt`
- `negative_prompt`
- `num_versions`
- `S_pos`

### 你要看什么

这是“生成清单”。后面 `04_run_recon_batch.py` 就按这里去执行。

---

## 2.6 04_run_recon_batch.py

### 作用
批量执行 ReCon 生成。

### 当前状态

脚本里现在是 **占位实现**，会把目标框画到图上，目的是先把整个工程链跑通。你后续需要把这里替换成真实的：

- ReCon
- ControlNet
- SAM
- Grounding DINO
- LoRA

实际推理调用。

### 它做什么

1. 读取 `recon_tasks.jsonl`；
2. 对每个任务生成多个版本；
3. 把输出图写到临时目录；
4. 输出 `manifest.jsonl`。

### 输入

- `recon_tasks.jsonl`
- 原始图像路径

### 输出

- `recon_outputs_tmp/{stage}/...jpg`
- `manifest.jsonl`

### 你要看什么

后面所有 QA 都基于这些临时输出。

---

## 2.7 05_qc_recon_outputs.py

### 作用
对 ReCon 输出做多维质检。

### 它做什么

1. 读取 `manifest.jsonl`；
2. 逐个样本计算：
   - `Q_gdino`
   - `Q_sam`
   - `Q_yolo`
   - `Q_diversity`
3. 合成 `Q_final`；
4. 根据 class-specific threshold 决定通过或拒绝；
5. 输出 `recon_qc_all.csv` 与 `recon_qc_pass.csv`。

### 输入

- `manifest.jsonl`
- 配置文件
- （真实环境下）GDINO / SAM / YOLO26 推理接口

### 输出

- `metadata/recon_qc_all.csv`
- `metadata/recon_qc_pass.csv`

### 关键字段

- `task_id`
- `class_id`
- `Q_gdino`
- `Q_sam`
- `Q_yolo`
- `Q_diversity`
- `Q_final`
- `pass_or_reject`
- `reject_reason`

### 你要看什么

这一步决定“哪些图最终能进训练集”。它是整个流程最关键的关口。

---

## 2.8 06_assemble_balance_dataset.py

### 作用
把通过 QA 的样本组装成最终增强子集。

### 它做什么

1. 读取 `recon_qc_pass.csv`；
2. 按类别配额、每图新增数、每源复用数做最终裁剪；
3. 把 pass 图像复制或链接到 `VisDrone_ReCon_Pass/`；
4. 写对应标签；
5. 输出最终增强集 metadata。

### 输入

- `recon_qc_pass.csv`
- pass 图像
- 原始标签信息

### 输出

- `VisDrone_ReCon_Pass/images/train/`
- `VisDrone_ReCon_Pass/labels/train/`
- `metadata/final_class_counts.csv`

### 你要看什么

这一步是真正落地成训练样本的地方。

---

## 2.9 07_make_split_lists.py

### 作用
生成训练清单。

### 它做什么

1. 写 `train_raw.txt`；
2. 写 `train_recon_pass.txt`；
3. 拼出 `train_union.txt`；
4. 写 `val_raw.txt`。

### 输入

- 原始数据集路径
- ReCon pass 子集路径

### 输出

- `split_files/train_raw.txt`
- `split_files/train_recon_pass.txt`
- `split_files/train_union.txt`
- `split_files/val_raw.txt`

### 你要看什么

训练时真正读的就是 `train_union.txt`。

---

## 3. Shell 脚本说明

## 3.1 08_build_dataset_mini.sh

### 作用
按 mini 阶段顺序执行 01–07。

### 它做什么

```text
实例库构建
→ 候选位置生成
→ ReCon 任务生成
→ 批量生成
→ 质检
→ 组装 mini 增强子集
→ 写 split 清单
```

### 适用场景

- 第一次验证流程；
- 只想先做小规模实验。

---

## 3.2 09_build_dataset_full.sh

### 作用
按 full 阶段顺序执行 01–07。

### 适用场景

- mini 已通过；
- 开始全量构建最终增强集。

---

## 3.3 10_write_data_yaml.sh

### 作用
根据 `train_union_{stage}.txt` 和 `val_raw.txt` 写训练用 YAML。

### 输出

例如：

- `VisDrone_ReCon_mini.yaml`
- `VisDrone_ReCon_full.yaml`

### 为什么要单独做

因为你可能反复切换：
- raw baseline
- raw + mini pass
- raw + full pass

单独写 yaml 最灵活。

---

## 3.4 11_train_mini.sh

### 作用
按你当前方案的训练设置跑 mini 训练。

### 典型用途

- 先看总体方向对不对；
- 看 Q_final 阈值要不要调；
- 看 C1/C2/C3 要不要调。

---

## 3.5 12_train_full.sh

### 作用
按同样配置跑 full 训练。

### 注意

只有 mini 阶段验证有效后才跑它。

---

## 3.6 13_run_mini_pipeline.sh

### 作用
一条命令跑完 mini：

```text
构建 mini 数据集
→ 写 yaml
→ 训练
```

### 适用场景

想一键跑通最小闭环时用它。

---

## 3.7 14_run_full_pipeline.sh

### 作用
一条命令跑完 full：

```text
构建 full 数据集
→ 写 yaml
→ 训练
```

### 注意

只在 mini 已有效的前提下使用。

---

## 4. 推荐运行顺序

## 4.1 最推荐的最小闭环

### 第一步：修改配置文件
先改 `00_config_example.yaml` 中的路径和规则。

### 第二步：跑 mini 数据集构建

```bash
bash sh/08_build_dataset_mini.sh
```

### 第三步：写 mini yaml

```bash
bash sh/10_write_data_yaml.sh mini
```

### 第四步：跑 mini 训练

```bash
bash sh/11_train_mini.sh
```

### 或者直接一条命令

```bash
bash sh/13_run_mini_pipeline.sh
```

---

## 4.2 full 阶段顺序

在 mini 有效后：

```bash
bash sh/09_build_dataset_full.sh
bash sh/10_write_data_yaml.sh full
bash sh/12_train_full.sh
```

或者：

```bash
bash sh/14_run_full_pipeline.sh
```

---

## 5. 最重要的输入 / 输出关系

## 5.1 从原始集到实例库

```text
原始图像 + 原始标签
→ 01_build_instance_bank.py
→ source_instances_pass.csv
```

## 5.2 从实例库到候选位置

```text
source_instances_pass.csv
→ 02_build_position_candidates.py
→ candidate_positions.csv
```

## 5.3 从候选位置到生成任务

```text
candidate_positions.csv
→ 03_make_recon_tasks.py
→ recon_tasks.jsonl
```

## 5.4 从生成任务到临时输出

```text
recon_tasks.jsonl
→ 04_run_recon_batch.py
→ recon_outputs_tmp/ + manifest.jsonl
```

## 5.5 从临时输出到 pass 样本

```text
manifest.jsonl
→ 05_qc_recon_outputs.py
→ recon_qc_pass.csv
→ 06_assemble_balance_dataset.py
→ VisDrone_ReCon_Pass/
```

## 5.6 从 pass 样本到训练清单

```text
VisDrone 原始集 + VisDrone_ReCon_Pass/
→ 07_make_split_lists.py
→ train_union.txt
```

## 5.7 从训练清单到训练

```text
train_union.txt + val_raw.txt
→ 10_write_data_yaml.sh
→ 11/12_train_*.sh
```

---

## 6. 当前脚本的限制

这套脚本已经把流程拆清楚了，但你要知道：

1. `04_run_recon_batch.py` 里的 ReCon 仍是占位实现，需要接真实推理；
2. `01_build_instance_bank.py` 里的 SAM 指标也还是占位接口，需要接真实 SAM；
3. `05_qc_recon_outputs.py` 里的 GDINO / YOLO26 也应接真实推理接口；
4. 目前脚本重点是 **把整条工程链跑通**，不是已经完成真实模型接入。

所以你可以把这套脚本理解成：

> **可运行的流程骨架 + 真实模型接口预留。**

---

## 7. 最后一条建议

你现在最合理的用法不是一上来就跑 full，而是：

1. 先改好 yaml；
2. 跑 mini pipeline；
3. 看 `source_instances_pass.csv`、`candidate_positions.csv`、`recon_qc_pass.csv`；
4. 人工抽查 6/7/8 三类 pass 图质量；
5. 如果质量过关，再进入 full。

一句话概括：

> **Python 脚本负责把“数据生产链”拆清楚，sh 脚本负责把“阶段性执行链”串起来；两者一起用，才是最适合你当前实验迭代方式的组织形式。**
