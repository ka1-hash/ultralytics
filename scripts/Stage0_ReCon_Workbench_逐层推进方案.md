# Stage 0：独立 ReCon Workbench 逐层推进方案

## 目标

在 `/home/shukang/project/recon_workbench/` 创建一个独立于 `YOLO26/` 核心项目的 ReCon 处理工作区。Stage 0 只做路径对齐、官方 ReCon 代码合并、依赖安装、完整 helper smoke test；不生成数据集、不训练模型、不改 YOLO26 核心代码。

## 输入目录

默认需要：

```bash
/home/shukang/project/upload/ReCon_A100_Standalone/
/home/shukang/project/upload/ReCon-master.zip
/home/shukang/project/upload/recon.zip
/home/shukang/project/datasets/VisDrone/
/home/shukang/project/YOLO26/   # 可选，仅做 symlink，不修改
```

## 输出目录

```bash
/home/shukang/project/recon_workbench/
├── assets/ReCon-master/         # hf_models + ckpts + 官方 ReCon pipelines/models/utils
├── assets/third_party/          # GroundingDINO + SAM 源码
├── toolkit/recon_balance678/    # 独立生成脚本，不写入 YOLO26
├── datasets_in/VisDrone -> /home/shukang/project/datasets/VisDrone
├── outputs/
├── logs/
├── env/recon_workbench.env
└── scripts/
```

## 执行

```bash
cd /home/shukang/project/upload
chmod +x stage0_recon_workbench_prepare_strict.sh
bash stage0_recon_workbench_prepare_strict.sh
```

如果环境已经装好：

```bash
SETUP_ENV=0 bash stage0_recon_workbench_prepare_strict.sh
```

如果先不加载 SD1.5/ControlNet：

```bash
RUN_PIPELINE_LOAD_TEST=0 bash stage0_recon_workbench_prepare_strict.sh
```

如果当前官方 ReConHelper 尚未 patch `perception_vocab`，但你只想先测试官方加载：

```bash
REQUIRE_VISDRONE_VOCAB=0 bash stage0_recon_workbench_prepare_strict.sh
```

正式用于 VisDrone 尾类 ReCon 时，必须让 `REQUIRE_VISDRONE_VOCAB=1` 通过。

## Stage 0 成功标准

`logs/stage0/stage0_smoke_test.log` 中必须出现：

```text
[SMOKE] required files OK
[SMOKE] groundingdino import OK
[SMOKE] segment_anything import OK
[SMOKE] ReConHelper import OK
[SMOKE] VisDrone perception vocab support OK
[SMOKE] SD1.5 + ControlNet + official ReCon pipeline load OK
[SMOKE] STAGE0 PASSED
```

## Stage 1 只做 10 张 smoke build

Stage 0 通过后，执行：

```bash
source /home/shukang/project/recon_workbench/env/recon_workbench.env
conda activate recon
bash /home/shukang/project/recon_workbench/scripts/stage1_smoke_build_10.sh
```

Stage 1 只生成 10 张，不允许直接全量生成。
