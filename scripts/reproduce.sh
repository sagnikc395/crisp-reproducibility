#!/usr/bin/env bash
# One command to reproduce Table 1 of the CRISP paper for a (model, domain) pair.
#
# It fetches the datasets into data/, evaluates the original model, trains CRISP
# and both baselines, and aggregates everything into artifacts/results/.
#
#   scripts/reproduce.sh configs/gemma2-2b_bio.yaml --local   # M-series Mac
#   scripts/reproduce.sh configs/gemma2-2b_bio.yaml           # full paper scale
#   scripts/reproduce.sh configs/smoke.yaml --local           # 1-minute sanity run
#
# Flags handled here (everything else is forwarded to `crisp`):
#   --local    Apple-silicon preset: MPS + float32, subsampled full-MMLU column
#   --fresh    re-run stages whose results already exist (default: resume)
#   --stages   comma-separated subset of original,crisp,rmu,elm
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
LOCAL=0
FRESH=0
STAGES="original,crisp,rmu,elm"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --local)  LOCAL=1; shift ;;
    --fresh)  FRESH=1; shift ;;
    --stages) STAGES="${2:?--stages needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *.yaml|*.yml) CONFIG="$1"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
: "${CONFIG:?usage: scripts/reproduce.sh <config.yaml> [--local] [--fresh] [extra crisp args]}"
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

# --- presets -----------------------------------------------------------------
OVERRIDES=()
if [ "${LOCAL}" = "1" ]; then
  # Apple silicon: MPS with float32 (bf16 optimiser math is unreliable there).
  # The full-MMLU utility column is 14k questions, which dominates every
  # evaluation; 2 per subject keeps all 57 subjects represented at ~1% of the
  # cost. WMDP and the in-domain MMLU columns stay at full size.
  OVERRIDES+=(-o model.device=mps -o model.dtype=float32 -o eval.mmlu_max_per_subject=2)
  # Safety net for any op Metal has not implemented; without it an unsupported
  # kernel aborts the run outright instead of running on the CPU.
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  echo "preset: --local (mps/float32, full-MMLU subsampled to 2 questions/subject)"
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
# Gated weights fail at download time, which on a laptop is several minutes into
# a multi-hour run. Check access up front instead.
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
echo "=== datasets -> data/ ================================================="
if [ "${HAS_LOCAL_CORPUS}" = "1" ]; then
  echo "config pins a local corpus; fetching MCQ benchmarks only"
  "${PY}" -m crisp fetch --domain "${DOMAIN}" --skip-corpora
else
  "${PY}" -m crisp fetch --domain "${DOMAIN}"
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

# --- 5. table ----------------------------------------------------------------
echo "=== results ==========================================================="
"${PY}" -m crisp report
echo "artifacts/results/: one JSON per run, plus summary.json and README.md"
