#!/usr/bin/env bash
# Analyze the 64-prompt × 4-sample full evaluation dumps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python

for n in base drgrpo6 answer6 process6 dense6 loop6; do
  dir="verl_impl/outputs/full_${n}"
  [ -d "$dir" ] || continue
  echo "=== full $n ==="
  PYTHONNOUSERSITE=1 "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "full_$n" \
    --json-out "$dir/summary.json" --save-groups "$dir/groups.json"
done

for n in drgrpo6 answer6 process6 dense6 loop6; do
  [ -f "verl_impl/outputs/full_${n}/groups.json" ] || continue
  PYTHONNOUSERSITE=1 "$PY" verl_impl/stats.py \
    --a "base=verl_impl/outputs/full_base/groups.json" \
    --b "$n=verl_impl/outputs/full_${n}/groups.json" \
    --json-out "verl_impl/outputs/full_${n}/stats_vs_base.json"
done

if [ -f "verl_impl/outputs/full_answer6/groups.json" ] && \
   [ -f "verl_impl/outputs/full_drgrpo6/groups.json" ]; then
  PYTHONNOUSERSITE=1 "$PY" verl_impl/stats.py \
    --a answer6=verl_impl/outputs/full_answer6/groups.json \
    --b drgrpo6=verl_impl/outputs/full_drgrpo6/groups.json \
    --json-out verl_impl/outputs/full_drgrpo6/stats_vs_answer6.json
fi
