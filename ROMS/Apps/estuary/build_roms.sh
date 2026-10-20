#!/bin/bash
#
# Build the ROMS executable for the micro-estuary application (MICRO_ESTUARY).
#
# The ROMS source and the build directories stay in the Linux home (compiling
# on /mnt/c is slow); only the application header is read from this folder.
#
# Usage:
#   ./build_roms.sh [--mixing my25|gls] [--mode serial|openmp|mpi] [--dyes]
#                   [--debug] [--noclean] [-j N] [--cpp "-DFLAG ..." --suffix NAME]
#
# Result: ${ROMS_BIN:-$HOME/roms/bin}/micro_estuary_<mixing>_<mode>[_dyes][_debug]
#
# Environment (optional):
#   ROMS_SRC    ROMS source tree        (default $HOME/roms/src)
#   ROMS_BUILD  build root              (default $HOME/roms/build)
#   ROMS_BIN    executables directory   (default $HOME/roms/bin)

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
ROMS_SRC="${ROMS_SRC:-$HOME/roms/src}"
ROMS_BUILD="${ROMS_BUILD:-$HOME/roms/build}"
ROMS_BIN="${ROMS_BIN:-$HOME/roms/bin}"

mixing=my25
mode=serial
dyes=no
debug=no
clean=yes
jobs=4
extra_cpp=""
suffix=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mixing)  mixing="$2"; shift 2 ;;
    --mode)    mode="$2";   shift 2 ;;
    --dyes)    dyes=yes;    shift ;;
    --debug)   debug=yes;   shift ;;
    --cpp)     extra_cpp="$2"; shift 2 ;;
    --suffix)  suffix="$2"; shift 2 ;;
    --noclean) clean=no;    shift ;;
    -j)        jobs="$2";   shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

case "$mixing" in my25|gls) ;; *) echo "--mixing must be my25 or gls" >&2; exit 1 ;; esac
case "$mode" in serial|openmp|mpi) ;; *) echo "--mode must be serial, openmp or mpi" >&2; exit 1 ;; esac

if [ ! -f "$ROMS_SRC/makefile" ]; then
  echo "ROMS source not found in $ROMS_SRC (run ROMS/setup/install_ubuntu.sh first)" >&2
  exit 1
fi

name="micro_estuary_${mixing}_${mode}"
export MY_CPP_FLAGS=""
if [ "$mixing" = gls ]; then MY_CPP_FLAGS="$MY_CPP_FLAGS -DUSE_GLS"; fi
if [ "$dyes" = yes ];  then MY_CPP_FLAGS="$MY_CPP_FLAGS -DUSE_DYES"; name="${name}_dyes"; fi
# extra CPP flags for tests (e.g. --cpp -DUSE_N2S2_HORAVG --suffix horavg)
if [ -n "$extra_cpp" ]; then MY_CPP_FLAGS="$MY_CPP_FLAGS $extra_cpp"; fi
if [ -n "$suffix" ];    then name="${name}_${suffix}"; fi

export ROMS_APPLICATION=MICRO_ESTUARY
export MY_HEADER_DIR="$APP_DIR"
export MY_ANALYTICAL_DIR="$APP_DIR"
export FORT=gfortran
export USE_NETCDF4=on
export USE_LARGE=on
export STATIC=on
export EXEC=on
unset USE_MPI USE_MPIF90 USE_OpenMP USE_DEBUG
case "$mode" in
  mpi)
    command -v mpif90 >/dev/null || { echo "mpif90 not found: install openmpi (see install_ubuntu.sh)" >&2; exit 1; }
    export USE_MPI=on USE_MPIF90=on which_MPI=openmpi ;;
  openmp)
    export USE_OpenMP=on ;;
esac
if [ "$debug" = yes ]; then export USE_DEBUG=on; name="${name}_debug"; fi

export BUILD_DIR="$ROMS_BUILD/$name"
export SCRATCH_DIR="$BUILD_DIR"
export BINDIR="$ROMS_BIN"
export BIN="$ROMS_BIN/$name"
mkdir -p "$BUILD_DIR" "$ROMS_BIN"

echo "ROMS source:       $ROMS_SRC"
echo "Application:       $ROMS_APPLICATION ($APP_DIR/micro_estuary.h)"
echo "CPP flags added:   ${MY_CPP_FLAGS:-none}"
echo "Build directory:   $BUILD_DIR"
echo "Executable:        $BIN"

cd "$ROMS_SRC"

# --dyes needs the ventilation-age patch (code guarded by AGE_SURFACE_RESET,
# so builds without it are unaffected)
if [ "$dyes" = yes ]; then
  patch="$APP_DIR/patches/age_surface_reset.patch"
  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"; echo "Applied $patch"
  elif git apply -R --check "$patch" 2>/dev/null; then
    echo "Patch already applied: $(basename "$patch")"
  else
    echo "Cannot apply $patch to $ROMS_SRC (source changed?)" >&2; exit 1
  fi
fi

if [ "$clean" = yes ]; then
  make clean > /dev/null
fi
make -j "$jobs"
echo "Built $BIN"
