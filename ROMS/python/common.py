"""
Shared paths and helpers for building the ROMS input files.

The model grids are generated once by GRIDATA/grid_IC.py (pygridgen, WSL env
`gridding4pom`) and saved as GRIDATA/grids/<domain>_grid.npz. Everything in
ROMS/python reads those files, so it only needs numpy/xarray/netCDF4 and runs
in any Python.

Array convention: the .npz arrays are (i, j) = (along channel, across channel),
i=0 at the mouth. ROMS arrays are (eta, xi) with xi along the channel, so they
are the transpose.
"""
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GRIDATA = REPO / 'GRIDATA'
ROMS_DIR = REPO / 'ROMS'
# input-file folder: ROMS/Data by default; ROMS_DATA (relative to ROMS/ or absolute) selects
# another one, e.g. ROMS_DATA=Data_N14 for the vertical-resolution test
DATA_DIR = ROMS_DIR / os.environ.get('ROMS_DATA', 'Data')
# grid folder in GRIDATA: grids by default (mouth = row 0, the 0.01 m bar); ROMS_GRIDS=grids_bar
# selects the closed-mouth grids of GRIDATA/add_sea_mouth.py (sea rows, then the raised crest)
GRIDS_DIR = GRIDATA / os.environ.get('ROMS_GRIDS', 'grids')
DOMAINS = ('Alexander', 'Meander', 'NoMeander')

# Passive tracers (ROMS inert tracers, NPT = 6, in this order), built with
# build_roms.sh --dyes (T_PASSIVE, AGE_MEAN, AGE_SURFACE_RESET). AGE_MEAN pairs
# them as (concentration c, age concentration a); mean age = a / c.
#   dye_01: river (fresh) water fraction: 0 at t=0, 1 in river water, 0 from the sea
#   dye_02: seawater fraction: 0 at t=0, 1 in water entering from the sea, 0 from the river
#   dye_03: unit tracer (1 everywhere); its age is the ventilation age, reset to 0
#           in water entering from the sea or river and in the surface layer.
#           It must stay the LAST pair: the surface reset acts on inert(NPT).
# Boundary and river files number them 01..06 in this order (dye_west_NN, river_dye_NN).
# columns: initial-file name, initial value, sea-boundary value, river value, long name, units
PASSIVE = (
    ('dye_01', 0.0, 0.0, 1.0, 'river (fresh) water fraction', 'nondimensional'),
    ('dye_01_age', 0.0, 0.0, 0.0, 'river water age concentration', 'second'),
    ('dye_02', 0.0, 1.0, 0.0, 'seawater fraction', 'nondimensional'),
    ('dye_02_age', 0.0, 0.0, 0.0, 'seawater age concentration', 'second'),
    ('dye_03', 1.0, 1.0, 1.0, 'unit tracer for the ventilation age', 'nondimensional'),
    ('dye_03_age', 0.0, 0.0, 0.0, 'ventilation age concentration (zero in the surface layer)', 'second'),
)

EARTH_RADIUS = 6371e3       # m
OMEGA = 7.2921e-5           # Earth rotation rate (1/s)


def load_grid(domain):
    '''Grid written by grid_IC.py, transposed to ROMS (eta, xi) order.'''
    g = np.load(GRIDS_DIR / f'{domain}_grid.npz')
    out = {k: g[k] for k in g.files}
    out['mouth_index'] = int(out.get('mouth_index', 0))
    for k in ('x', 'y', 'lon', 'lat', 'h'):
        out[k] = np.ascontiguousarray(out[k].T)
    for k in ('T', 'S'):                       # (i, j, k) -> (k, eta, xi)
        out[k] = np.ascontiguousarray(np.transpose(out[k], (2, 1, 0)))
    return out


def mouth_index(grid):
    '''xi index of the mouth (the bar or crest row): the grid.nc attribute or npz entry, else 0.'''
    if isinstance(grid, dict):
        return int(grid.get('mouth_index', 0))
    return int(grid.attrs.get('mouth_index', 0))


def estuary_slice(grid, ndim=2):
    '''(eta, xi) index of the estuary cells: interior rows, from the cell after the mouth to the
    cell before the river-end row; ndim=3 prepends the vertical axis.'''
    sl = (slice(1, -1), slice(mouth_index(grid) + 1, -1))
    return (slice(None),) * (ndim - 2) + sl


def domain_dir(domain):
    path = DATA_DIR / domain
    path.mkdir(parents=True, exist_ok=True)
    return path


def rho_to_u(a):
    '''Average two neighbouring rho points along xi (eta, xi) -> (eta, xi-1).'''
    return 0.5 * (a[:, :-1] + a[:, 1:])


def rho_to_v(a):
    '''Average two neighbouring rho points along eta (eta, xi) -> (eta-1, xi).'''
    return 0.5 * (a[:-1, :] + a[1:, :])


def rho_to_psi(a):
    '''Average four neighbouring rho points -> (eta-1, xi-1).'''
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])
