#!/bin/bash
#
# Prepare an Ubuntu (or WSL Ubuntu) machine to build and run the ROMS
# micro-estuary simulations. Safe to re-run: every step checks what is already
# there and only adds what is missing.
#
#   1. system packages (asks for sudo only if something is missing)
#   2. ROMS source:   $ROMS_SRC   (default ~/roms/src, from github.com/myroms/roms)
#   3. executables:   $ROMS_BIN   (default ~/roms/bin) via ROMS/Apps/estuary/build_roms.sh
#   4. output folder: <modelling>/ROMS_runs
#
# Usage:
#   ROMS/setup/install_ubuntu.sh [--with-mpi] [--skip-build] [--python-env]
#
#   --with-mpi     also install OpenMPI and build the MPI executables
#   --skip-build   do not compile ROMS
#   --python-env   create a conda env "roms-estuary" (needs conda/miniforge) with the
#                  packages for the grid generation and analysis scripts
#
# Tested on WSL Ubuntu 24.04 (gfortran 13, netCDF-Fortran 4.5.4).

set -e

with_mpi=no
build=yes
python_env=no
for a in "$@"; do
  case "$a" in
    --with-mpi)   with_mpi=yes ;;
    --skip-build) build=no ;;
    --python-env) python_env=yes ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown option $a" >&2; exit 1 ;;
  esac
done

SETUP="$(cd "$(dirname "$0")" && pwd)"
ROMS_DIR="$(dirname "$SETUP")"
REPO="$(dirname "$ROMS_DIR")"
ROMS_SRC="${ROMS_SRC:-$HOME/roms/src}"
ROMS_RUNS="${ROMS_RUNS:-$(dirname "$REPO")/ROMS_runs}"

# ---- 1. system packages -------------------------------------------------
need=()
check() { command -v "$1" >/dev/null 2>&1 || need+=("$2"); }
check gfortran gfortran
check make make
check perl perl
check git git
check python3 python3
check nf-config libnetcdff-dev
check ncdump netcdf-bin
check ncks nco
if [ "$with_mpi" = yes ]; then
  check mpif90 libopenmpi-dev
  check mpirun openmpi-bin
fi
if [ ${#need[@]} -gt 0 ]; then
  echo "Installing missing packages: ${need[*]}"
  sudo apt-get update
  sudo apt-get install -y "${need[@]}"
else
  echo "System packages: all present"
fi

# ---- 2. ROMS source -------------------------------------------------------
if [ -f "$ROMS_SRC/makefile" ]; then
  echo "ROMS source: $ROMS_SRC ($(git -C "$ROMS_SRC" log -1 --format='%h %ad' --date=short 2>/dev/null))"
else
  mkdir -p "$(dirname "$ROMS_SRC")"
  git clone --depth 1 https://github.com/myroms/roms.git "$ROMS_SRC"
fi

# ---- 3. executables -----------------------------------------------------
if [ "$build" = yes ]; then
  python3 "$ROMS_DIR/Apps/estuary/make_in_template.py" "$ROMS_SRC/ROMS/External/roms_upwelling.in"
  modes=(serial)
  [ "$with_mpi" = yes ] && modes+=(mpi)
  for mode in "${modes[@]}"; do
    for mixing in my25 gls; do
      log="$HOME/roms/build_${mixing}_${mode}.log"
      echo "Building $mixing/$mode (log: $log)"
      ROMS_SRC="$ROMS_SRC" bash "$ROMS_DIR/Apps/estuary/build_roms.sh" --mixing "$mixing" --mode "$mode" -j "$(nproc)" > "$log" 2>&1
      tail -1 "$log"
    done
  done
fi

# ---- 4. output folder and Python ----------------------------------------
mkdir -p "$ROMS_RUNS"
echo "Simulation output: $ROMS_RUNS"

if [ "$python_env" = yes ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create -y -n roms-estuary -c conda-forge python=3.11 numpy pandas xarray netcdf4 \
        matplotlib pyyaml openpyxl geopandas pygridgen
  else
    echo "conda not found: install Miniforge first (https://github.com/conda-forge/miniforge)" >&2
  fi
fi

echo "Done. Next: python ROMS/python/make_inputs.py (if the inputs are not there yet),"
echo "then ROMS/scripts/run_experiment.sh ROMS/scripts/experiments/ini_fill.cfg"
