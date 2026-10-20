#!/bin/bash
#
# Run one ROMS simulation of the micro-estuary application.
#
# Usage:
#   ./run_one.sh [--dry-run] DOMAIN EXPERIMENT RUN [KEY=VALUE ...]
#
#   DOMAIN      Alexander | Meander | NoMeander
#   EXPERIMENT  experiment name (folder under ROMS_RUNS)
#   RUN         run name (folder under the experiment)
#   KEY=VALUE   overrides of ROMS/scripts/defaults.cfg (e.g. DAYS=2 AKS_BAK=1e-6
#               RIVERS=on RIVER_SCALE=4 DATA=Data_N20u INI_FILE=ini_raw.nc
#               WIND_FILE=wind_w2.nc EXEC=micro_estuary_my25_serial_dyes_wind)
#
# Creates $ROMS_RUNS/EXPERIMENT/RUN/ with copies of the input files, the filled
# roms.in, run_meta.json and the ROMS log (roms.log); output NetCDF files are
# written there too. --dry-run prepares the folder without running ROMS.
#
# Environment (optional):
#   ROMS_RUNS   output root (default: <modelling>/ROMS_runs, next to this repository)
#   ROMS_BIN    executables (default: $HOME/roms/bin)
#   ROMS_SRC    ROMS source, for varinfo.yaml (default: $HOME/roms/src)

set -e

dry_run=no
if [ "$1" = "--dry-run" ]; then dry_run=yes; shift; fi
if [ $# -lt 3 ]; then sed -n '2,22p' "$0"; exit 1; fi

DOMAIN="$1"; EXPERIMENT="$2"; RUN="$3"; shift 3

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ROMS_DIR="$(dirname "$SCRIPTS")"
REPO="$(dirname "$ROMS_DIR")"
ROMS_RUNS="${ROMS_RUNS:-$(dirname "$REPO")/ROMS_runs}"
ROMS_BIN="${ROMS_BIN:-$HOME/roms/bin}"
DATA_SUB=Data                                   # input folder, DATA=... selects another one
INI_FILE=ini.nc                                 # day-1 state of a static run (ini_from_run.py);
                                                # INI_FILE=ini_raw.nc: the prescribed fields
for a in "$@"; do case "$a" in DATA=*) DATA_SUB="${a#DATA=}" ;; INI_FILE=*) INI_FILE="${a#INI_FILE=}" ;; esac; done
DATA="$ROMS_DIR/$DATA_SUB/$DOMAIN"
OUTDIR="$ROMS_RUNS/$EXPERIMENT/$RUN"

for f in grid.nc "$INI_FILE" bry.nc summary.json; do
  [ -f "$DATA/$f" ] || { echo "Missing $DATA/$f (run ROMS/python/make_inputs.py, then ini_from_run.py for ini.nc)" >&2; exit 1; }
done

mkdir -p "$OUTDIR"
cp "$DATA/grid.nc" "$DATA/bry.nc" "$OUTDIR/"
cp "$DATA/$INI_FILE" "$OUTDIR/ini.nc"
[ -f "$DATA/rivers.nc" ] && cp "$DATA/rivers.nc" "$OUTDIR/"

read -r EXEC NPROC RIVERS < <(python3 "$SCRIPTS/fill_in.py" "$DOMAIN" "$EXPERIMENT" "$RUN" "$OUTDIR" "$@")

# RIVER_SCALE=k multiplies the river transport of the copied rivers.nc (flow experiments)
RIVER_SCALE=$(python3 -c "import json; print(json.load(open('$OUTDIR/run_meta.json'))['parameters'].get('RIVER_SCALE', '1'))")
if [ "$RIVER_SCALE" != "1" ] && [ -f "$OUTDIR/rivers.nc" ]; then
  command -v ncap2 >/dev/null || { echo "RIVER_SCALE needs NCO (ncap2)" >&2; exit 1; }
  ncap2 -O -s "river_transport=river_transport*$RIVER_SCALE" "$OUTDIR/rivers.nc" "$OUTDIR/rivers.nc"
  echo "River flow:  x$RIVER_SCALE"
fi

# WIND_FILE=<file in the input folder>: surface stress for builds with -DUSE_WIND_FILE
WIND_FILE=$(python3 -c "import json; print(json.load(open('$OUTDIR/run_meta.json'))['parameters'].get('WIND_FILE', 'none'))")
if [ "$WIND_FILE" != "none" ]; then
  [ -f "$DATA/$WIND_FILE" ] || { echo "Missing $DATA/$WIND_FILE (ROMS/python/roms_wind.py --out $WIND_FILE)" >&2; exit 1; }
  [[ "$EXEC" == *wind* ]] || { echo "WIND_FILE needs an executable built with --cpp -DUSE_WIND_FILE (got $EXEC)" >&2; exit 1; }
  cp "$DATA/$WIND_FILE" "$OUTDIR/wind.nc"
  echo "Wind:        $WIND_FILE"
elif [[ "$EXEC" == *wind* ]]; then
  echo "Executable $EXEC reads wind.nc; set WIND_FILE" >&2; exit 1
fi
echo "Run folder:  $OUTDIR"
echo "Executable:  $ROMS_BIN/$EXEC (processes: $NPROC, $RIVERS)"

if [ "$dry_run" = yes ]; then
  echo "Dry run: roms.in written, ROMS not started."
  exit 0
fi

[ -x "$ROMS_BIN/$EXEC" ] || { echo "Executable not found: $ROMS_BIN/$EXEC (build with ROMS/Apps/estuary/build_roms.sh)" >&2; exit 1; }

cd "$OUTDIR"
start=$(date +%s)
if [ "$NPROC" -gt 1 ] && [[ "$EXEC" == *_mpi* ]]; then
  mpirun -np "$NPROC" "$ROMS_BIN/$EXEC" roms.in > roms.log 2>&1 || status=$?
else
  "$ROMS_BIN/$EXEC" < roms.in > roms.log 2>&1 || status=$?
fi
status=${status:-0}
end=$(date +%s)
echo "{\"exit_status\": $status, \"wall_seconds\": $((end - start))}" > run_status.json
if grep -q "ROMS: DONE" roms.log; then
  echo "Finished in $((end - start)) s: $OUTDIR"
else
  echo "ROMS did not finish normally (exit $status); see $OUTDIR/roms.log" >&2
  tail -20 roms.log >&2
  exit 1
fi
