#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-}"
gpu=0
workers=4
check_only=false
no_amp=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --framework) framework="$2"; shift 2;;
    --method) method="$2"; shift 2;;
    --dataset) dataset="$2"; shift 2;;
    --data-root) data_root="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    --checkpoint) checkpoint="$2"; shift 2;;
    --result-json) result_json="$2"; shift 2;;
    --num-workers) workers="$2"; shift 2;;
    --check-only) check_only=true; shift;;
    --no-amp) no_amp=true; shift;;
    -h|--help)
      echo "Usage: bash test.sh --framework mvfa|madclip|iqeclip --method baseline|sggp --dataset DATASET --data-root PATH [options]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

case "${framework:-}" in
  mvfa) python_bin="${MVFA_PYTHON:-python}" ;;
  madclip) python_bin="${MADCLIP_PYTHON:-python}" ;;
  iqeclip) python_bin="${IQECLIP_PYTHON:-python}" ;;
  *) echo "framework is required" >&2; exit 2 ;;
esac

: "${method:?--method is required}"
: "${dataset:?--dataset is required}"
: "${data_root:?--data-root is required}"
checkpoint="${checkpoint:-$root/checkpoints/$framework/$method/$dataset.pth}"
[[ -f "$checkpoint" ]] || { echo "checkpoint not found: $checkpoint" >&2; exit 1; }
[[ -f "$root/checkpoints/pretrained/ViT-L-14-336px.pt" ]] || { echo "pretrained weight not found" >&2; exit 1; }

command=(
  "$python_bin" "$root/frameworks/$framework/test.py"
  --method "$method"
  --dataset "$dataset"
  --data-root "$data_root"
  --checkpoint "$checkpoint"
  --gpu "$gpu"
  --num-workers "$workers"
)

if [[ -n "${result_json:-}" ]]; then
  command+=(--result-json "$result_json")
fi
if $check_only; then
  command+=(--check-only)
fi
if $no_amp; then
  command+=(--no-amp)
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "${command[@]}"
