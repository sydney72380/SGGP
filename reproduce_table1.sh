#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-}"
gpus="0,1"
out="$root/results/table1_reproduction"
workers=4
check_only=false
no_amp=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) data_root="$2"; shift 2;;
    --gpus) gpus="$2"; shift 2;;
    --output) out="$2"; shift 2;;
    --num-workers) workers="$2"; shift 2;;
    --check-only) check_only=true; shift;;
    --no-amp) no_amp=true; shift;;
    -h|--help)
      echo "Usage: bash reproduce_table1.sh --data-root PATH [--gpus 0,1] [--output DIR] [--check-only]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

: "${data_root:?--data-root is required}"
[[ "$out" = /* ]] || out="$root/$out"
IFS=',' read -ra gpu_list <<< "$gpus"
mkdir -p "$out/tasks" "$out/logs"

frameworks=(mvfa madclip iqeclip)
methods=(baseline sggp)
datasets=(Chest Brain Liver Retina_RESC Histopathology Retina_OCT2017)
pids=()
names=()
number=0

wait_batch() {
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "done: ${names[$i]}"
    else
      echo "failed: ${names[$i]} (see $out/logs/${names[$i]}.log)" >&2
      return 1
    fi
  done
  pids=()
  names=()
}

for dataset in "${datasets[@]}"; do
  for framework in "${frameworks[@]}"; do
    for method in "${methods[@]}"; do
      name="${framework}_${method}_${dataset}"
      gpu="${gpu_list[$((number % ${#gpu_list[@]}))]}"
      args=(
        --framework "$framework"
        --method "$method"
        --dataset "$dataset"
        --data-root "$data_root"
        --gpu "$gpu"
        --num-workers "$workers"
        --result-json "$out/tasks/$name.json"
      )
      if $check_only; then args+=(--check-only); fi
      if $no_amp; then args+=(--no-amp); fi
      echo "gpu $gpu: $name"
      bash "$root/test.sh" "${args[@]}" >"$out/logs/$name.log" 2>&1 &
      pids+=("$!")
      names+=("$name")
      number=$((number + 1))
      if ((${#pids[@]} == ${#gpu_list[@]})); then
        wait_batch
      fi
    done
  done
done
if ((${#pids[@]} > 0)); then
  wait_batch
fi

python_bin="${ORCHESTRATOR_PYTHON:-${MVFA_PYTHON:-python}}"
"$python_bin" - "$out" "$check_only" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
check_only = sys.argv[2] == "true"
datasets = ("Retina_OCT2017", "Histopathology", "Chest", "Brain", "Liver", "Retina_RESC")
items = (
    ("MVFA", "mvfa", "baseline"),
    ("MVFA+SGGP", "mvfa", "sggp"),
    ("MadCLIP", "madclip", "baseline"),
    ("MadCLIP+SGGP", "madclip", "sggp"),
    ("IQE-CLIP", "iqeclip", "baseline"),
    ("IQE-CLIP+SGGP", "iqeclip", "sggp"),
)
expected = (
    (99.57, 83.58, 82.46, 90.94, 96.89, 86.94, 99.57, 95.86, 99.16, 89.89, 98.54),
    (99.67, 83.84, 83.27, 91.74, 97.75, 87.13, 99.71, 96.29, 99.03, 90.32, 98.83),
    (99.63, 82.43, 82.70, 92.64, 97.31, 83.42, 99.55, 94.28, 98.56, 89.18, 98.47),
    (99.84, 83.04, 82.56, 92.85, 97.33, 83.46, 99.53, 94.29, 98.69, 89.34, 98.52),
    (98.83, 74.01, 79.72, 81.27, 97.70, 62.74, 99.47, 93.71, 98.72, 81.71, 98.63),
    (98.91, 77.52, 77.76, 85.19, 97.77, 63.60, 99.53, 94.30, 98.75, 82.88, 98.68),
)
records = {}
for label, framework, method in items:
    for dataset in datasets:
        path = out / "tasks" / f"{framework}_{method}_{dataset}.json"
        if not path.exists():
            raise SystemExit(f"missing result: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        status = "checkpoint_load_ok" if check_only else "ok"
        if record.get("status") != status:
            raise SystemExit(f"bad result: {path}")
        records[(framework, method, dataset)] = record

if check_only:
    print("Checkpoint load verification passed: 36/36")
    print("TABLE1_CHECKPOINT_LOAD_PASS")
    raise SystemExit

columns = ("OCT", "HIS", "Chest", "Brain", "Brain pAUC", "Liver", "Liver pAUC", "RESC", "RESC pAUC", "Mean AUC", "Mean pAUC")
print("\t".join(("Method", *columns)))
mismatches = []
for row_index, (label, framework, method) in enumerate(items):
    rows = [records[(framework, method, dataset)] for dataset in datasets]
    auc = {dataset: round(float(row["auc"]) * 100, 2) for dataset, row in zip(datasets, rows)}
    pauc = {dataset: round(float(records[(framework, method, dataset)]["pauc"]) * 100, 2) for dataset in ("Brain", "Liver", "Retina_RESC")}
    values = (
        auc["Retina_OCT2017"], auc["Histopathology"], auc["Chest"], auc["Brain"], pauc["Brain"],
        auc["Liver"], pauc["Liver"], auc["Retina_RESC"], pauc["Retina_RESC"],
        sum(auc.values()) / 6, sum(pauc.values()) / 3,
    )
    print("\t".join((label, *(f"{value:.2f}" for value in values))))
    for column, actual, wanted in zip(columns, values, expected[row_index]):
        if f"{actual:.2f}" != f"{wanted:.2f}":
            mismatches.append((label, column, wanted, actual))

print(f"Tasks: 36; summary rows: 6; mismatches: {len(mismatches)}")
if mismatches:
    for row in mismatches:
        print("MISMATCH", *row)
    raise SystemExit(2)
print("TABLE1_REPRODUCTION_PASS")
PY
