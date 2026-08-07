#!/usr/bin/env bash
# One command to reproduce Table 1 of the CRISP paper for a (model, domain) pair.
#
# It evaluates the original model, trains CRISP and both baselines, aggregates
# everything into artifacts/results/ and renders artifacts/figures/.
#
# This runs on a CUDA GPU -- in practice the Colab notebook
# (notebooks/crisp_colab.ipynb), which calls exactly this script. Locally the
# only thing worth running is the dataset fetch (`python -m crisp fetch`) and
# the smoke config.
#
#   scripts/reproduce.sh configs/gemma2-2b_bio.yaml        # full run on a GPU
#   scripts/reproduce.sh configs/smoke.yaml                # 1-minute sanity run, CPU is fine
#
# Flags handled here (everything else is forwarded to `crisp`):
#   --full-mmlu  keep the ~14k-question general-MMLU column at full size
#                (default: 2 questions/subject, all 57 subjects represented)
#   --no-fetch   assume data/ is already populated (e.g. mounted from Drive)
#   --fresh      re-run stages whose results already exist (default: resume)
#   --stages     comma-separated subset of original,crisp,rmu,elm
#
# Credentials come from .env at the repo root (HF_TOKEN for the gated Gemma
# weights and bio forget corpus). The fluency/concept columns are scored by a
# local Qwen3 rater downloaded on first use; pass --no-judge to skip it and
# leave those two columns blank.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${PY:-${ROOT}/.venv/bin/python}"
[ -x "${PY}" ] || PY="python3"

CONFIG=""
FULL_MMLU=0
NO_FETCH=0
FRESH=0
STAGES="original,crisp,rmu,elm"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --full-mmlu) FULL_MMLU=1; shift ;;
    --no-fetch)  NO_FETCH=1; shift ;;
    --fresh)     FRESH=1; shift ;;
    --stages)    STAGES="${2:?--stages needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *.yaml|*.yml) CONFIG="$1"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
: "${CONFIG:?usage: scripts/reproduce.sh <config.yaml> [--full-mmlu] [--fresh] [extra crisp args]}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }

NAME="$(basename "${CONFIG}" .yaml)"

if ! "${PY}" -c "import crisp" 2>/dev/null; then
  echo "cannot import crisp with ${PY}." >&2
  echo "  Set the project up first:  uv sync --extra dev" >&2
  echo "  Or point PY at another interpreter:  PY=/path/to/python $0 ..." >&2
  exit 1
fi

read -r DOMAIN SPLIT HAS_LOCAL_CORPUS <<EOF
$("${PY}" - "${CONFIG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
data, ev = cfg.get("data") or {}, cfg.get("eval") or {}
print(data.get("domain", "bio"), ev.get("split", "test"),
      1 if data.get("target_corpus") else 0)
PYEOF
)
EOF

# --- device / cost presets ---------------------------------------------------
# bf16 on Ampere and newer; float32 on pre-Ampere cards (a T4 has no native
# bf16, and training here runs without a gradient scaler, so float16 would give
# silent NaNs instead of a clean OOM). CPU falls back to float32 too.
OVERRIDES=()
eval "$("${PY}" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(0)
    dtype = "bfloat16" if major >= 8 else "float32"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'DEVICE=cuda; DTYPE={dtype}; GPU="{name} ({vram:.0f} GB, sm_{major}x)"')
else:
    print('DEVICE=cpu; DTYPE=float32; GPU="no CUDA device"')
PYEOF
)"
OVERRIDES+=(-o "model.device=${DEVICE}" -o "model.dtype=${DTYPE}")
echo "device: ${GPU} -> ${DEVICE}/${DTYPE}"

if [ "${FULL_MMLU}" = "0" ]; then
  # The general-MMLU utility column is 14k questions and otherwise dominates
  # every stage. 2 per subject keeps all 57 subjects represented at ~1% of the
  # cost. WMDP and the in-domain MMLU columns -- the ones the paper's claims
  # rest on -- always stay at full size.
  OVERRIDES+=(-o eval.mmlu_max_per_subject=2)
  echo "general-MMLU column subsampled to 2 questions/subject (--full-mmlu to disable)"
fi

# ${arr[@]+...} keeps an empty array from tripping `set -u` on bash 3.2 (macOS).
run_crisp() {
  echo "+ crisp $*"
  "${PY}" -m crisp "$@" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"} ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
}

wants() { case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# A stage is done when its result file exists; that is what `crisp report` reads.
done_already() {
  [ "${FRESH}" = "0" ] && [ -f "artifacts/results/${1}__${SPLIT}.json" ]
}

skip_or_run() {
  local run_name="$1"; shift
  if done_already "${run_name}"; then
    echo "--- ${run_name}: already in artifacts/results, skipping (use --fresh to redo)"
    return 0
  fi
  "$@"
}

# --- preflight ---------------------------------------------------------------
# Gated weights fail at download time, which is several minutes into a
# multi-hour run. Check access up front instead.
"${PY}" - "${CONFIG}" <<'PYEOF' || exit 1
import sys, yaml
from crisp.utils import hf_token   # loads .env

name = ((yaml.safe_load(open(sys.argv[1])) or {}).get("model") or {}).get("name", "")
if name.startswith(("google/", "meta-llama/")):
    if not hf_token():
        sys.exit(f"preflight: {name} is gated but no HF_TOKEN found.\n"
                 f"  Accept the licence at https://huggingface.co/{name}, then\n"
                 f"  echo 'HF_TOKEN=hf_...' >> .env")
    # repo_info() answers for any public metadata even without access, so use
    # auth_check(), which is the call that actually tests the gate.
    from huggingface_hub import auth_check
    try:
        auth_check(name, token=hf_token())
    except Exception as exc:
        sys.exit(f"preflight: HF_TOKEN cannot access {name} ({type(exc).__name__}).\n"
                 f"  Accept the licence at https://huggingface.co/{name} and retry.")
print(f"preflight ok: {name or 'local model'}")
PYEOF

# --- 0. datasets -------------------------------------------------------------
# Normally fetched once on the laptop and carried up (Drive, or a re-fetch here);
# `crisp fetch` is a no-op for files already on disk, so this is cheap either way.
if [ "${NO_FETCH}" = "1" ]; then
  echo "=== datasets: --no-fetch, using what is already in data/ ==============="
else
  echo "=== datasets -> data/ ================================================="
  if [ "${HAS_LOCAL_CORPUS}" = "1" ]; then
    echo "config pins a local corpus; fetching MCQ benchmarks only"
    "${PY}" -m crisp fetch --domain "${DOMAIN}" --skip-corpora
  else
    "${PY}" -m crisp fetch --domain "${DOMAIN}"
  fi
fi

# --- 1-4. the four models ----------------------------------------------------
if wants original; then
  echo "=== original model ===================================================="
  skip_or_run "${NAME}_original" run_crisp eval -c "${CONFIG}" --run-name "${NAME}_original"
fi

if wants crisp; then
  echo "=== CRISP ============================================================="
  skip_or_run "${NAME}_crisp" run_crisp train -c "${CONFIG}" --run-name "${NAME}_crisp"
fi

if wants rmu; then
  echo "=== RMU baseline ======================================================"
  skip_or_run "${NAME}_rmu" run_crisp baseline -c "${CONFIG}" --method rmu --run-name "${NAME}"
fi

if wants elm; then
  echo "=== ELM baseline ======================================================"
  skip_or_run "${NAME}_elm" run_crisp baseline -c "${CONFIG}" --method elm --run-name "${NAME}"
fi

# --- 5. table and figures ----------------------------------------------------
echo "=== results ==========================================================="
"${PY}" -m crisp report
"${PY}" -m crisp plots
echo "artifacts/results/: one JSON per run, plus summary.json and README.md"
echo "artifacts/figures/: metric bars, forget/retain trade-off, training curves"
