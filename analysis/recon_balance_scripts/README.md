# ReCon 类别平衡脚本包

这是一套与《ReCon_类别平衡最终全量说明方案_终稿》配套的脚本模板，目标是把方案拆成可执行的工程骨架。

## 目录

- `00_config_example.yaml`：配置示例
- `common.py`：公共函数
- `01_build_instance_bank.py`：构建实例库与源样本初筛
- `02_build_position_candidates.py`：生成候选位置并打分
- `03_make_recon_tasks.py`：生成 ReCon 任务清单
- `04_run_recon_batch.py`：批量执行 ReCon 任务（预留模型接入接口）
- `05_qc_recon_outputs.py`：对 ReCon 输出做多维质检
- `06_assemble_balance_dataset.py`：组装最终增强数据集
- `07_make_split_lists.py`：生成 raw/recon/union 训练清单

## 说明

1. 这些脚本是**按最终方案规则整理的可落地模板**，已经把：
   - S2/S3/S4 统计与评分入口
   - C1/C2/C3 的配额与上限逻辑
   - mini → full 的两阶段流程
   - 磁盘友好的 union txt 组织方式
   都做进去了。
2. 与 SAM / Grounding DINO / ReCon / YOLO26 的真实推理调用，当前保留了清晰的接口函数；你可以在对应位置替换为项目里的真实调用。
3. 推荐执行顺序：

```bash
python 01_build_instance_bank.py --config 00_config_example.yaml
python 02_build_position_candidates.py --config 00_config_example.yaml
python 03_make_recon_tasks.py --config 00_config_example.yaml --stage mini
python 04_run_recon_batch.py --config 00_config_example.yaml --stage mini
python 05_qc_recon_outputs.py --config 00_config_example.yaml --stage mini
python 06_assemble_balance_dataset.py --config 00_config_example.yaml --stage mini
python 07_make_split_lists.py --config 00_config_example.yaml --stage mini
```

## 两阶段建议

### mini 阶段
- 每图最多新增 1 个目标
- 每类只做少量高质量 pass
- 用于阈值校准与快速验证

### full 阶段
- 允许 class 6/8 在高质量条件下每图最多 2 个
- class 7 仍建议每图最多 1 个
- 只在 mini 验证通过后启用

## 磁盘建议

- 原始数据集不要复制
- 只保存 `ReCon pass` 最终图像与标签
- 训练通过 `train_union.txt = train_raw.txt + train_recon_pass.txt` 联合读取
- `recon_outputs_tmp/`、未通过 QA 的候选图建议及时删除


## Shell 包装脚本

- `sh/08_build_dataset_mini.sh`：mini 阶段一键构建增强集
- `sh/09_build_dataset_full.sh`：full 阶段一键构建增强集
- `sh/10_write_data_yaml.sh`：根据 split txt 生成训练用 data yaml
- `sh/11_train_mini.sh`：mini 阶段训练模板
- `sh/12_train_full.sh`：full 阶段训练模板
- `sh/13_run_mini_pipeline.sh`：mini 阶段从构建到训练一键跑通
- `sh/14_run_full_pipeline.sh`：full 阶段从构建到训练一键跑通

### 推荐执行顺序

```bash
# 1) mini 先验证
bash sh/08_build_dataset_mini.sh 00_config_example.yaml
bash sh/10_write_data_yaml.sh mini /home/shukang/datasets/split_files /home/shukang/datasets/VisDrone_ReCon_Data_mini.yaml
bash sh/11_train_mini.sh

# 2) mini 通过后，再做 full
bash sh/09_build_dataset_full.sh 00_config_example.yaml
bash sh/10_write_data_yaml.sh full /home/shukang/datasets/split_files /home/shukang/datasets/VisDrone_ReCon_Data_full.yaml
bash sh/12_train_full.sh
```
