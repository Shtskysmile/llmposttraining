#!/usr/bin/env bash
set -euo pipefail
cd /root/work/tiankuan/shtskysmile/llmposttraining
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python

for name in base process2 process6 answer6 dense6 drgrpo6 format6 loop6; do
  dir="verl_impl/outputs/screen_${name}"
  [ -d "$dir" ] || continue
  echo "=== screen $name ==="
  "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "$name" \
    --json-out "$dir/summary.json" --save-groups "$dir/groups.json"
done

for name in process6 answer6 dense6 drgrpo6 format6 loop6; do
  dir="verl_impl/outputs/${name}/rollout"
  [ -d "$dir" ] || continue
  echo "=== dynamics $name ==="
  "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "$name" --per-step \
    --json-out "$dir/dynamics.json"
done
