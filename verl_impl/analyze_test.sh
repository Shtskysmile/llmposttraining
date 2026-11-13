#!/usr/bin/env bash
# Analyze the 96-prompt held-out word-problem test dumps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python

for n in base answer6 drgrpo6 process6; do
  dir="verl_impl/outputs/test_${n}"
  [ -d "$dir" ] || continue
  echo "=== test $n ==="
  PYTHONNOUSERSITE=1 "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "test_$n" \
    --json-out "$dir/summary.json" --save-groups "$dir/groups.json"
done

for n in answer6 drgrpo6 process6; do
  [ -f "verl_impl/outputs/test_${n}/groups.json" ] || continue
  PYTHONNOUSERSITE=1 "$PY" verl_impl/stats.py \
    --a "base=verl_impl/outputs/test_base/groups.json" \
    --b "$n=verl_impl/outputs/test_${n}/groups.json" \
    --json-out "verl_impl/outputs/test_${n}/stats_vs_base.json"
done
