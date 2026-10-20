"""
Closed mouth: add sea cells and a raised sandbar crest to the grids of grid_IC.py.

In grids/<domain>_grid.npz the mouth is row 0, a 0.01 m bar that is also the open boundary.
ROMS takes the depth of the face between that row and the first estuary cell as the mean
of the two, ~1.8 m, so a reversing wind drives a two-layer exchange through it and seawater
enters at sea level 0.

This script writes grids_bar/<domain>_grid.npz with, along the channel from the open
(western) boundary:
    rows 0 .. n_sea-1   sea, depth sea_depth (2 m), open boundary at row 0
    row  n_sea          the sandbar crest (the old row 0), h = -crest_height (above sea level)
    rows n_sea+1 ..     the estuary, unchanged (old rows 1 ..)
The sea rows are copies of the old row 0 shifted seaward, normal to the mouth row, by the length of
the first centreline cell (one shift for the whole row, so the sea cells are rectangles).
With the crest dry (zeta + h <= Dcrit), ROMS WET_DRY lets no water off the crest, so seawater
enters only if the sea level exceeds crest_height + Dcrit.

The npz gets mouth_index = n_sea (the crest row); ROMS/python reads it.

Usage: python add_sea_mouth.py [--sea-rows 2] [--sea-depth 2.0] [--crest-height 0.10]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = ('Alexander', 'Meander', 'NoMeander')
SEA_SALT = 39.5


def seaward_step(x, y):
    '''one shift (dx, dy) for the whole mouth row: normal to the line through the mouth row,
    pointing seaward, as long as the first centreline cell; the sea cells are translations of
    the mouth row, so they cannot cross and are orthogonal to it'''
    c = x.shape[1] // 2
    tx, ty = x[0, -1] - x[0, 0], y[0, -1] - y[0, 0]
    nx, ny = ty, -tx
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    sx, sy = x[0, c] - x[1, c], y[0, c] - y[1, c]        # seaward along the centreline
    if nx * sx + ny * sy < 0:
        nx, ny = -nx, -ny
    step = np.hypot(sx, sy)
    return nx * step, ny * step


def extend(a, n_sea, step):
    '''prepend n_sea copies of row 0 shifted by k * step (k = n_sea .. 1)'''
    new = [a[0] + (n_sea - k) * step for k in range(n_sea)]
    return np.concatenate([np.stack(new), a], axis=0)


def add_mouth(src, n_sea=2, sea_depth=2.0, crest_height=0.10):
    g = dict(np.load(src))
    out = dict(g)
    dx, dy = seaward_step(g['x'], g['y'])
    # lon/lat shifted with the same fractions of the first cell (linear over ~30 m)
    c = g['x'].shape[1] // 2
    A = np.array([[g['x'][0, c] - g['x'][1, c], g['x'][0, -1] - g['x'][0, 0]],
                  [g['y'][0, c] - g['y'][1, c], g['y'][0, -1] - g['y'][0, 0]]])
    a, b = np.linalg.solve(A, [dx, dy])                  # step = a * centre step + b * mouth-row vector
    for k in ('x', 'y', 'lon', 'lat'):
        v = g[k]
        step = a * (v[0, c] - v[1, c]) + b * (v[0, -1] - v[0, 0])
        out[k] = extend(v, n_sea, step)
    h = g['h'].copy()                                 # positive depth, (i, j)
    h[0, :] = -crest_height                           # the crest, banks included
    out['h'] = np.concatenate([np.full((n_sea, h.shape[1]), sea_depth), h], axis=0)
    for k, fill in (('T', None), ('S', SEA_SALT)):    # POM-level fields, kept for shape consistency
        sea = np.repeat(g[k][:1], n_sea, axis=0)
        if fill is not None:
            sea[:] = fill
        out[k] = np.concatenate([sea, g[k]], axis=0)
    out.update(mouth_index=n_sea, sea_rows=n_sea, sea_depth=sea_depth, crest_height=crest_height,
               mouth_depth=-crest_height)
    return out


def main(n_sea=2, sea_depth=2.0, crest_height=0.10, src_dir='grids', out_dir='grids_bar'):
    src_dir, out_dir = os.path.join(HERE, src_dir), os.path.join(HERE, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for d in DOMAINS:
        out = add_mouth(os.path.join(src_dir, f'{d}_grid.npz'), n_sea, sea_depth, crest_height)
        np.savez(os.path.join(out_dir, f'{d}_grid.npz'), **out)
        dx = np.hypot(np.diff(out['x'][:n_sea + 2, 3]), np.diff(out['y'][:n_sea + 2, 3]))
        print(f"{d}: {out['h'].shape[0]} x {out['h'].shape[1]}, crest row {n_sea} h = {out['h'][n_sea, 3]:+.2f} m, "
              f"sea {sea_depth} m, first spacings {np.round(dx, 1)} m")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Add sea cells and a raised crest at the mouth')
    p.add_argument('--sea-rows', type=int, default=2)
    p.add_argument('--sea-depth', type=float, default=2.0)
    p.add_argument('--crest-height', type=float, default=0.10, help='crest above sea level (m)')
    a = p.parse_args()
    main(a.sea_rows, a.sea_depth, a.crest_height)
