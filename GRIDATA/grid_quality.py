"""
Grid quality and time-step diagnostics for the model grids.

Reads grids/<domain>_grid.npz (written by grid_IC.py) or, with --ic, the
ALON/ALAT/H blocks of POM IC_*.dat files, and reports per domain:
    - external-mode (barotropic) CFL time step per cell,
          dt = 1 / ( sqrt(g h) * sqrt(1/dx^2 + 1/dy^2) )
      with dx, dy the cell spacings (centred differences, as in pom08 file2ic)
    - advective limit at 1 m/s, min(dx, dy) / 1 m/s
    - orthogonality (angle between the i and j grid lines) and aspect ratio
and the ratio of the global minimum dt to the 1st-percentile dt, which shows
whether a few cells limit the time step.

Usage:
    python grid_quality.py                       # new grids (grids/*.npz)
    python grid_quality.py --ic IC_Alexander.dat IC_Meander.dat ...
"""
import argparse
import os

import numpy as np
import pandas as pd

G = 9.81
H_MIN = 0.05          # wetting/drying critical depth (POM hc, ROMS Dcrit)


def read_ic(filename, IM=200, JM=7):
    '''Read lon, lat and H from a POM IC_*.dat file (H positive down).'''
    blocks = {}
    cur = None
    with open(filename) as f:
        for line in f:
            s = line.strip()
            try:
                v = float(s)
            except ValueError:
                cur = s
                blocks[cur] = []
                continue
            blocks[cur].append(v)
    shape = lambda a: np.array(a).reshape(JM, IM).T
    lon, lat, h = shape(blocks['ALON']), shape(blocks['ALAT']), -shape(blocks['H'])
    r = 6371e3
    x = np.radians(lon) * r * np.cos(np.radians(lat.mean()))
    y = np.radians(lat) * r
    return x, y, h


def quality(x, y, h):
    '''Per-cell diagnostics on the interior points (i, j = 1..N-2).'''
    dx = 0.5 * np.hypot(x[2:, 1:-1] - x[:-2, 1:-1], y[2:, 1:-1] - y[:-2, 1:-1])
    dy = 0.5 * np.hypot(x[1:-1, 2:] - x[1:-1, :-2], y[1:-1, 2:] - y[1:-1, :-2])
    hh = np.maximum(h[1:-1, 1:-1], H_MIN)
    dt = 1.0 / (np.sqrt(G * hh) * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    # angle between i- and j-lines at each node of the interior
    ax_, ay_ = x[2:, 1:-1] - x[:-2, 1:-1], y[2:, 1:-1] - y[:-2, 1:-1]
    bx_, by_ = x[1:-1, 2:] - x[1:-1, :-2], y[1:-1, 2:] - y[1:-1, :-2]
    cosang = np.abs(ax_ * bx_ + ay_ * by_) / (np.hypot(ax_, ay_) * np.hypot(bx_, by_))
    angle = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    return dict(dx=dx, dy=dy, dt=dt, angle=angle, adv=np.minimum(dx, dy) / 1.0)


def summarise(name, q):
    dt = q['dt']
    imin = np.unravel_index(np.argmin(dt), dt.shape)
    return {
        'domain': name,
        'dt_min_s': dt.min(),
        'dt_p01_s': np.percentile(dt, 1),
        'dt_median_s': np.median(dt),
        'dt_min/p01': dt.min() / np.percentile(dt, 1),
        'limiting_cell_ij': f'({imin[0] + 1},{imin[1] + 1})',
        'dx_min_m': q['dx'].min(), 'dx_median_m': np.median(q['dx']),
        'dy_min_m': q['dy'].min(), 'dy_median_m': np.median(q['dy']),
        'adv_dt_min_s': q['adv'].min(),
        'angle_min_deg': q['angle'].min(),
        'angle_p01_deg': np.percentile(q['angle'], 1),
        'aspect_max': np.max(np.maximum(q['dx'], q['dy']) / np.minimum(q['dx'], q['dy'])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ic', nargs='*', help='POM IC_*.dat files instead of grids/*.npz')
    parser.add_argument('--label', default='', help='label added to the domain names')
    parser.add_argument('--out', default=None, help='CSV file for the summary table')
    args = parser.parse_args()

    rows = []
    if args.ic:
        for f in args.ic:
            x, y, h = read_ic(f)
            rows.append(summarise(os.path.basename(f) + args.label, quality(x, y, h)))
    else:
        for name in ('Alexander', 'Meander', 'NoMeander'):
            g = np.load(f'./grids/{name}_grid.npz')
            rows.append(summarise(name + args.label, quality(g['x'], g['y'], g['h'])))
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    if args.out:
        table.to_csv(args.out, index=False)
    return table


if __name__ == '__main__':
    main()
