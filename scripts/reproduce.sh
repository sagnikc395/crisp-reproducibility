#!/usr/bin/env bash
# Reproduce Table 1 of the CRISP paper for one (model, domain) pair.
#
#   scripts/reproduce.sh configs/gemma2-2b_cyber.yaml
#
# Requires HF_TOKEN (gated Gemma/Llama weights, gated WMDP bio-forget corpus)
# and, for the fluency/concept columns, ANTHROPIC_API_KEY.
set -euo pipefail

CONFIG="${1:?usage: scripts/reproduce.sh <config.yaml> [extra crisp args...]}"
shift || true
NAME="$(basename "${CONFIG}" .yaml)"
PY="${PY:-.venv/bin/python}"

echo "=== 1/4  original model ==============================================="
"${PY}" -m crisp eval -c "${CONFIG}" --run-name "${NAME}_original" \
  --out "outputs/eval/${NAME}_original.json" "$@"

echo "=== 2/4  CRISP ========================================================"
"${PY}" -m crisp train -c "${CONFIG}" --run-name "${NAME}_crisp" "$@"

echo "=== 3/4  RMU baseline ================================================="
"${PY}" -m crisp baseline -c "${CONFIG}" --method rmu --run-name "${NAME}" "$@"

echo "=== 4/4  ELM baseline ================================================="
"${PY}" -m crisp baseline -c "${CONFIG}" --method elm --run-name "${NAME}" "$@"

echo
echo "results:"
find outputs -name "eval_test.json" -o -name "${NAME}_original.json" | sort
