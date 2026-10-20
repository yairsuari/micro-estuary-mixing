#!/bin/bash
#
# Run all simulations listed in an experiment file, several at a time.
#
# Usage:
#   ./run_experiment.sh [--dry-run] experiments/NAME.cfg
#
# Experiment file format:
#   EXPERIMENT=name          experiment folder under ROMS_RUNS
#   PARALLEL=3               number of runs at the same time (default 1)
#   COMMON KEY=VALUE ...     overrides applied to every run (optional, repeatable)
#   RUN DOMAIN [KEY=VALUE ...]
#                            one line per run: run name, domain, run-specific overrides
# Lines starting with # are comments.

set -e

dry=""
if [ "$1" = "--dry-run" ]; then dry="--dry-run"; shift; fi
cfg="$1"
[ -f "$cfg" ] || { sed -n '2,15p' "$0"; exit 1; }

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
experiment=""
parallel=1
common=()
runs=()

while read -r line; do
  line="${line%%#*}"
  [ -z "${line// }" ] && continue
  case "$line" in
    EXPERIMENT=*) experiment="${line#EXPERIMENT=}"; experiment="${experiment// }" ;;
    PARALLEL=*)   parallel="${line#PARALLEL=}"; parallel="${parallel// }" ;;
    COMMON\ *)    read -ra kv <<< "${line#COMMON }"; common+=("${kv[@]}") ;;
    *)            runs+=("$line") ;;
  esac
done < "$cfg"

[ -n "$experiment" ] || { echo "EXPERIMENT= missing in $cfg" >&2; exit 1; }
echo "Experiment $experiment: ${#runs[@]} runs, $parallel at a time"

failed=0
running=0
for spec in "${runs[@]}"; do
  read -ra f <<< "$spec"
  run="${f[0]}"; domain="${f[1]}"; extra=("${f[@]:2}")
  echo "--> $run ($domain) ${common[*]} ${extra[*]}"
  "$SCRIPTS/run_one.sh" $dry "$domain" "$experiment" "$run" "${common[@]}" "${extra[@]}" \
      > "/tmp/${experiment}_${run}.out" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$parallel" ]; then
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
  fi
done
while [ "$running" -gt 0 ]; do
  wait -n || failed=$((failed + 1))
  running=$((running - 1))
done

for spec in "${runs[@]}"; do
  read -ra f <<< "$spec"
  echo "== ${f[0]}"; tail -3 "/tmp/${experiment}_${f[0]}.out"
done
echo "Experiment $experiment finished, $failed failed."
[ "$failed" -eq 0 ]
