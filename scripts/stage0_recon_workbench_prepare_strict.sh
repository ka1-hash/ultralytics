#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 0: ReCon A100 Workbench Prepare + Full-Helper Smoke Test
# Purpose:
#   Build an independent ReCon workspace outside YOLO26 core project.
#   It only aligns paths, merges official ReCon code/assets, installs deps,
#   and runs strict smoke tests. It does NOT generate datasets and does NOT train.
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/home/shukang/project}"
UPLOAD_ROOT="${UPLOAD_ROOT:-$PROJECT_ROOT/upload}"
WORKBENCH="${WORKBENCH:-$PROJECT_ROOT/recon_workbench}"
YOLO_ROOT="${YOLO_ROOT:-$PROJECT_ROOT/YOLO26}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$PROJECT_ROOT/datasets/VisDrone}"
STANDALONE_ROOT="${STANDALONE_ROOT:-$UPLOAD_ROOT/ReCon_A100_Standalone}"
RECON_MASTER_ZIP="${RECON_MASTER_ZIP:-$UPLOAD_ROOT/ReCon-master.zip}"
RECON_ZIP="${RECON_ZIP:-$UPLOAD_ROOT/recon.zip}"
CONDA_ENV="${CONDA_ENV:-recon}"
SETUP_ENV="${SETUP_ENV:-1}"
INSTALL_TORCH_IF_MISSING="${INSTALL_TORCH_IF_MISSING:-1}"
RUN_PIPELINE_LOAD_TEST="${RUN_PIPELINE_LOAD_TEST:-1}"
REQUIRE_VISDRONE_VOCAB="${REQUIRE_VISDRONE_VOCAB:-1}"

RECON_REPO="$WORKBENCH/assets/ReCon-master"
THIRD_PARTY="$WORKBENCH/assets/third_party"
TOOLKIT="$WORKBENCH/toolkit/recon_balance678"
LOG_DIR="$WORKBENCH/logs/stage0"
REPORT="$WORKBENCH/STAGE0_REPORT.md"

log() { echo -e "[STAGE0] $*"; }
fail() { echo -e "[STAGE0][FAILED] $*" >&2; exit 1; }
need_dir() { [[ -d "$1" ]] || fail "Missing directory: $1"; }
need_file() { [[ -f "$1" ]] || fail "Missing file: $1"; }

log "PROJECT_ROOT=$PROJECT_ROOT"
log "UPLOAD_ROOT=$UPLOAD_ROOT"
log "WORKBENCH=$WORKBENCH"
log "STANDALONE_ROOT=$STANDALONE_ROOT"
log "RECON_MASTER_ZIP=$RECON_MASTER_ZIP"
log "RECON_ZIP=$RECON_ZIP"

need_dir "$STANDALONE_ROOT"
need_dir "$VISDRONE_ROOT"
[[ -d "$YOLO_ROOT" ]] || log "WARNING: YOLO_ROOT not found yet: $YOLO_ROOT ; Stage0 can still prepare ReCon assets."

mkdir -p "$RECON_REPO" "$THIRD_PARTY" "$TOOLKIT" "$LOG_DIR" "$WORKBENCH"/{env,scripts,outputs,tmp,datasets_in}

# -----------------------------
# 1) Copy model assets
# -----------------------------
log "Copying hf_models and ckpts into independent workbench..."
need_dir "$STANDALONE_ROOT/hf_models"
need_dir "$STANDALONE_ROOT/ckpts"
rsync -a --delete "$STANDALONE_ROOT/hf_models/" "$RECON_REPO/hf_models/"
rsync -a --delete "$STANDALONE_ROOT/ckpts/" "$RECON_REPO/ckpts/"

# third_party may either live in standalone or official package; prefer standalone if available
if [[ -d "$STANDALONE_ROOT/third_party" ]]; then
  log "Copying third_party from ReCon_A100_Standalone..."
  rsync -a --delete "$STANDALONE_ROOT/third_party/" "$THIRD_PARTY/"
else
  log "WARNING: no standalone third_party found. Will rely on ReCon-master.zip if it contains needed code."
fi

# -----------------------------
# 2) Merge official ReCon code
# -----------------------------
if [[ -f "$RECON_MASTER_ZIP" ]]; then
  log "Extracting official ReCon-master.zip..."
  TMP_RECON="$WORKBENCH/tmp/recon_master_extract"
  rm -rf "$TMP_RECON"
  mkdir -p "$TMP_RECON"
  unzip -q -o "$RECON_MASTER_ZIP" -d "$TMP_RECON"
  OFFICIAL_ROOT="$(find "$TMP_RECON" -type f -path '*/pipelines/recon_helper.py' -printf '%h\n' | sed 's#/pipelines$##' | head -n 1 || true)"
  [[ -n "$OFFICIAL_ROOT" ]] || fail "Cannot find official ReCon root containing pipelines/recon_helper.py in $RECON_MASTER_ZIP"
  log "Official ReCon root detected: $OFFICIAL_ROOT"
  for d in pipelines models utils; do
    [[ -d "$OFFICIAL_ROOT/$d" ]] || fail "Official ReCon missing directory: $d"
    rsync -a --delete "$OFFICIAL_ROOT/$d/" "$RECON_REPO/$d/"
  done
  for f in deepcache_extension.py generate.py requirements.txt README.md; do
    [[ -f "$OFFICIAL_ROOT/$f" ]] && cp -f "$OFFICIAL_ROOT/$f" "$RECON_REPO/" || true
  done
else
  log "WARNING: ReCon-master.zip not found. Trying to use pipelines/models/utils from standalone."
  for d in pipelines models utils; do
    [[ -d "$STANDALONE_ROOT/$d" ]] || fail "No official ReCon code found. Missing $d in standalone and no ReCon-master.zip."
    rsync -a --delete "$STANDALONE_ROOT/$d/" "$RECON_REPO/$d/"
  done
fi

# -----------------------------
# 3) Install project-side generation toolkit without touching YOLO26 core
# -----------------------------
log "Installing recon_balance678 toolkit into independent workbench..."
if [[ -f "$RECON_ZIP" ]]; then
  TMP_TOOL="$WORKBENCH/tmp/recon_zip_extract"
  rm -rf "$TMP_TOOL"
  mkdir -p "$TMP_TOOL"
  unzip -q -o "$RECON_ZIP" -d "$TMP_TOOL"
  # Copy all likely useful project folders into toolkit; do not touch YOLO26.
  if [[ -d "$TMP_TOOL/analysis/recon_balance678" ]]; then
    mkdir -p "$TOOLKIT/analysis"
    rsync -a --delete "$TMP_TOOL/analysis/recon_balance678/" "$TOOLKIT/analysis/recon_balance678/"
  fi
  if [[ -d "$TMP_TOOL/configs" ]]; then
    rsync -a --delete "$TMP_TOOL/configs/" "$TOOLKIT/configs/"
  fi
  if [[ -d "$TMP_TOOL/scripts" ]]; then
    rsync -a --delete "$TMP_TOOL/scripts/" "$TOOLKIT/scripts/"
  fi
  if [[ -d "$TMP_TOOL/recipes" ]]; then
    rsync -a --delete "$TMP_TOOL/recipes/" "$TOOLKIT/recipes/"
  fi
else
  log "WARNING: recon.zip not found. Falling back to standalone/src and standalone/recipes."
  [[ -d "$STANDALONE_ROOT/src/recon_balance678" ]] || fail "Missing recon toolkit: neither recon.zip nor standalone/src/recon_balance678 exists."
  rsync -a --delete "$STANDALONE_ROOT/src/recon_balance678/" "$TOOLKIT/"
  [[ -d "$STANDALONE_ROOT/recipes" ]] && rsync -a --delete "$STANDALONE_ROOT/recipes/" "$TOOLKIT/recipes/" || true
fi

# convenience symlinks
ln -sfn "$VISDRONE_ROOT" "$WORKBENCH/datasets_in/VisDrone"
[[ -d "$YOLO_ROOT" ]] && ln -sfn "$YOLO_ROOT" "$WORKBENCH/yolo_project" || true

# -----------------------------
# 4) Write env file
# -----------------------------
cat > "$WORKBENCH/env/recon_workbench.env" <<EOF
export PROJECT_ROOT="$PROJECT_ROOT"
export WORKBENCH="$WORKBENCH"
export RECON_REPO="$RECON_REPO"
export THIRD_PARTY="$THIRD_PARTY"
export TOOLKIT="$TOOLKIT"
export YOLO_ROOT="$YOLO_ROOT"
export VISDRONE_ROOT="$VISDRONE_ROOT"
export PYTHONPATH="$RECON_REPO:$THIRD_PARTY/GroundingDINO-main:$THIRD_PARTY/segment-anything-main:$TOOLKIT:$PYTHONPATH"
export HF_HOME="$WORKBENCH/cache/huggingface"
export TRANSFORMERS_CACHE="$WORKBENCH/cache/huggingface"
export DIFFUSERS_CACHE="$WORKBENCH/cache/huggingface"
export TORCH_HOME="$WORKBENCH/cache/torch"
EOF
# replace placeholder char for literal PYTHONPATH fallback safely
python - <<'PY' "$WORKBENCH/env/recon_workbench.env"
from pathlib import Path
p=Path(__import__('sys').argv[1])
s=p.read_text()
s=s.replace(':\x7f$PYTHONPATH', ':${PYTHONPATH:-}')
p.write_text(s)
PY

# -----------------------------
# 5) Create strict smoke test script
# -----------------------------
cat > "$WORKBENCH/scripts/stage0_smoke_test.py" <<'PY'
import os
import sys
import inspect
from pathlib import Path

workbench = Path(os.environ.get("WORKBENCH", "/home/shukang/project/recon_workbench"))
recon_repo = Path(os.environ.get("RECON_REPO", workbench / "assets" / "ReCon-master"))
third_party = Path(os.environ.get("THIRD_PARTY", workbench / "assets" / "third_party"))
run_pipeline = os.environ.get("RUN_PIPELINE_LOAD_TEST", "1") == "1"
require_vocab = os.environ.get("REQUIRE_VISDRONE_VOCAB", "1") == "1"

sys.path.insert(0, str(recon_repo))
sys.path.insert(0, str(third_party / "GroundingDINO-main"))
sys.path.insert(0, str(third_party / "segment-anything-main"))

required = [
    recon_repo / "hf_models/stable-diffusion-v1-5/model_index.json",
    recon_repo / "hf_models/stable-diffusion-v1-5/unet/diffusion_pytorch_model.fp16.safetensors",
    recon_repo / "hf_models/stable-diffusion-v1-5/vae/diffusion_pytorch_model.fp16.safetensors",
    recon_repo / "hf_models/stable-diffusion-v1-5/text_encoder/model.fp16.safetensors",
    recon_repo / "hf_models/sd-controlnet-canny/config.json",
    recon_repo / "hf_models/sd-controlnet-canny/diffusion_pytorch_model.safetensors",
    recon_repo / "ckpts/sam_vit_h_4b8939.pth",
    recon_repo / "ckpts/grounding-dino-tiny/config.json",
    recon_repo / "ckpts/grounding-dino-tiny/model.safetensors",
    recon_repo / "pipelines/pipeline_controlnet_recon.py",
    recon_repo / "pipelines/recon_helper.py",
    recon_repo / "models/attention_processor.py",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("[SMOKE][FAILED] missing required files:")
    for m in missing:
        print("  -", m)
    raise SystemExit(2)
print("[SMOKE] required files OK")

try:
    import torch
    print("[SMOKE] torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[SMOKE] gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("[SMOKE][FAILED] torch import failed:", repr(e))
    raise

try:
    import groundingdino
    print("[SMOKE] groundingdino import OK")
except Exception as e:
    print("[SMOKE][FAILED] groundingdino import failed:", repr(e))
    raise

try:
    import segment_anything
    print("[SMOKE] segment_anything import OK")
except Exception as e:
    print("[SMOKE][FAILED] segment_anything import failed:", repr(e))
    raise

try:
    from pipelines.recon_helper import ReConHelper
    print("[SMOKE] ReConHelper import OK")
    sig = inspect.signature(ReConHelper.__init__)
    print("[SMOKE] ReConHelper signature:", sig)
    if require_vocab and "perception_vocab" not in sig.parameters:
        print("[SMOKE][FAILED] ReConHelper does not accept perception_vocab.")
        print("This means VisDrone tail-class full helper is not ready yet.")
        print("Patch pipelines/recon_helper.py or use a wrapper that provides VisDrone perception vocabulary.")
        raise SystemExit(3)
    print("[SMOKE] VisDrone perception vocab support OK")
except SystemExit:
    raise
except Exception as e:
    print("[SMOKE][FAILED] ReConHelper import/signature check failed:", repr(e))
    raise

if run_pipeline:
    try:
        import torch
        from diffusers import ControlNetModel, DDIMScheduler
        from pipelines.pipeline_controlnet_recon import StableDiffusionControlNetImg2ImgPipeline
        from compel import Compel
        sd15_dir = recon_repo / "hf_models/stable-diffusion-v1-5"
        control_dir = recon_repo / "hf_models/sd-controlnet-canny"
        controlnet = ControlNetModel.from_pretrained(str(control_dir), torch_dtype=torch.float16, local_files_only=True)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(sd15_dir),
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        _ = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        print("[SMOKE] SD1.5 + ControlNet + official ReCon pipeline load OK")
    except Exception as e:
        print("[SMOKE][FAILED] pipeline load failed:", repr(e))
        raise
else:
    print("[SMOKE] pipeline load skipped by RUN_PIPELINE_LOAD_TEST=0")

print("[SMOKE] STAGE0 PASSED")
PY

# -----------------------------
# 6) Create Stage1 smoke command helper, but do not execute
# -----------------------------
cat > "$WORKBENCH/scripts/stage1_smoke_build_10.sh" <<'BASH2'
#!/usr/bin/env bash
set -euo pipefail
source /home/shukang/project/recon_workbench/env/recon_workbench.env
cd "$TOOLKIT"

# Locate builder in independent toolkit.
BUILDER="analysis/recon_balance678/build_recon_balance678_dataset.py"
if [[ ! -f "$BUILDER" ]]; then
  echo "[STAGE1][FAILED] Cannot find builder at $TOOLKIT/$BUILDER" >&2
  exit 1
fi

CONFIG="configs/recon_balance678.json"
if [[ ! -f "$CONFIG" ]]; then
  # fallback for nested config exported by standalone package
  CONFIG="configs/recon_balance678/recon_balance678.json"
fi
[[ -f "$CONFIG" ]] || { echo "[STAGE1][FAILED] Cannot find recon_balance678 config" >&2; exit 1; }

python "$BUILDER" \
  --source-images-train "$VISDRONE_ROOT/images/train" \
  --source-labels-train "$VISDRONE_ROOT/labels/train" \
  --source-images-val "$VISDRONE_ROOT/images/val" \
  --source-labels-val "$VISDRONE_ROOT/labels/val" \
  --recon-repo "$RECON_REPO" \
  --config "$CONFIG" \
  --out-root "$WORKBENCH/outputs/VisDrone_ReCon_Smoke_A100" \
  --dataset-name "VisDrone_ReCon_Smoke_A100" \
  --device cuda \
  --seed 42 \
  --cfg-scale 4.0 \
  --strength 1.0 \
  --num-inference-steps 25 \
  --num-cache-steps 5 \
  --det-steps 0.75 0.5 0.25 0.1 \
  --min-short-side 12 \
  --max-neighbor-boxes 3 \
  --window-margin-ratio 0.60 \
  --min-boundary-px 4 \
  --recon-diversity 0.0 \
  --use-full-helper \
  --save-debug \
  --smoke-test \
  --smoke-max-images 10
BASH2
chmod +x "$WORKBENCH/scripts/stage1_smoke_build_10.sh"

# -----------------------------
# 7) Optional env install
# -----------------------------
if [[ "$SETUP_ENV" == "1" ]]; then
  log "Setting up conda env: $CONDA_ENV"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -n "$CONDA_ENV" python=3.10 -y >/dev/null 2>&1 || true
    conda activate "$CONDA_ENV"
  else
    fail "conda not found. Set SETUP_ENV=0 if env is already prepared."
  fi
  if ! python - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
  then
    if [[ "$INSTALL_TORCH_IF_MISSING" == "1" ]]; then
      log "Installing torch cu121 because torch import failed..."
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
      fail "torch not installed and INSTALL_TORCH_IF_MISSING=0"
    fi
  fi
  log "Installing ReCon lightweight dependencies..."
  pip install -U pip
  pip install diffusers==0.28.0 transformers==4.48.0 accelerate==0.30.0 safetensors
  pip install opencv-python Pillow numpy scipy tqdm matplotlib einops compel==2.0.3 pycocotools jsonlines scikit-image
  pip install -e "$THIRD_PARTY/GroundingDINO-main"
  pip install -e "$THIRD_PARTY/segment-anything-main"
else
  log "SETUP_ENV=0, skip dependency installation."
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || log "WARNING: cannot activate $CONDA_ENV now. Smoke test may fail if env is not active."
  fi
fi

# -----------------------------
# 8) Run smoke test
# -----------------------------
log "Running Stage0 smoke test..."
source "$WORKBENCH/env/recon_workbench.env"
export RUN_PIPELINE_LOAD_TEST="$RUN_PIPELINE_LOAD_TEST"
export REQUIRE_VISDRONE_VOCAB="$REQUIRE_VISDRONE_VOCAB"
python "$WORKBENCH/scripts/stage0_smoke_test.py" 2>&1 | tee "$LOG_DIR/stage0_smoke_test.log"

# -----------------------------
# 9) Report
# -----------------------------
DU_WORKBENCH="$(du -sh "$WORKBENCH" | awk '{print $1}')"
DU_RECON="$(du -sh "$RECON_REPO" | awk '{print $1}')"
DU_THIRD="$(du -sh "$THIRD_PARTY" | awk '{print $1}')"
DU_TOOLKIT="$(du -sh "$TOOLKIT" | awk '{print $1}')"
cat > "$REPORT" <<EOF
# Stage 0 ReCon Workbench Report

- WORKBENCH: $WORKBENCH
- RECON_REPO: $RECON_REPO
- THIRD_PARTY: $THIRD_PARTY
- TOOLKIT: $TOOLKIT
- VISDRONE_ROOT: $VISDRONE_ROOT
- YOLO_ROOT: $YOLO_ROOT
- Size WORKBENCH: $DU_WORKBENCH
- Size ReCon-master: $DU_RECON
- Size third_party: $DU_THIRD
- Size toolkit: $DU_TOOLKIT
- Smoke log: $LOG_DIR/stage0_smoke_test.log
- Stage1 smoke script: $WORKBENCH/scripts/stage1_smoke_build_10.sh

## Result

Stage 0 completed. If the smoke log contains `STAGE0 PASSED`, the independent ReCon workbench is ready for Stage 1 smoke dataset build.
EOF

log "Stage0 report: $REPORT"
log "Stage-0 completed successfully. Next step: bash $WORKBENCH/scripts/stage1_smoke_build_10.sh"
