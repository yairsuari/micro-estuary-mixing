"""
ROMS vertical (s-coordinate) grid, fitted to the POM sigma levels.

POM uses fixed sigma fractions from depth(KB) in GRIDATA/grid_IC.py
(log-linear-log: finer near the surface and the bottom). ROMS uses
    z = zeta + (zeta + h) * (hc*s + h*C(s)) / (hc + h)     (Vtransform = 2)
with C(s) from Vstretching = 4 (Shchepetkin 2010):
    Csur = (1 - cosh(theta_s*s)) / (cosh(theta_s) - 1)
    C    = (exp(theta_b*Csur) - 1) / (1 - exp(-theta_b))
When hc is much smaller than h, z/h = C(s): the levels are fixed fractions of
the depth everywhere, as in POM. hc is therefore set small (Tcline = hc).

theta_s and theta_b are fitted (least squares) so that C at the ROMS
interfaces matches the POM interfaces z(k). The parameters are written to
ROMS/Data/vgrid.yaml, which the IC builder and the .in template both read.

--n N keeps the fitted theta_s/theta_b (the same C(s) shape) but uses N layers.
--vstretching/--theta-s/--theta-b set the stretching directly instead of the
POM fit, e.g. uniform layers (fixed fractions of the depth):
    --vstretching 1 --theta-s 0 --theta-b 0
The POM fit refines the bottom strongly (theta_b = 3.8), which makes the grid
stiff (Haney rx1) when N is increased.

Usage: python roms_vgrid.py [--kb 8] [--hc 0.01] [--n N]
                            [--vstretching 1 --theta-s 0 --theta-b 0]
"""
import argparse
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import DATA_DIR, GRIDATA

sys.path.insert(0, str(GRIDATA))
from grid_IC import depth           # noqa: E402  POM vertical levels

VGRID_FILE = DATA_DIR / 'vgrid.yaml'

# Default vertical grid (decided 2026-10-03): 10 layers, mild surface
# refinement (Song and Haidvogel stretching, theta_s = 1, theta_b = 0). At 3.5 m depth the layers are
# 30 cm at the top to 44 cm at the bottom; maximum Haney number rx1 2.1 on the Alexander grid.
DEFAULT = {'N': 10, 'Vstretching': 1, 'theta_s': 1.0, 'theta_b': 0.0}


def s_levels(N):
    '''ROMS s at interfaces (N+1, -1..0) and at layer centres (N).'''
    s_w = (np.arange(N + 1) - N) / N
    s_r = (np.arange(1, N + 1) - N - 0.5) / N
    return s_w, s_r


def stretching(s, theta_s, theta_b, vstretching=4):
    '''ROMS stretching function C(s) (Vstretching 1 and 4, as in ROMS set_scoord.F).'''
    if vstretching == 1:                 # Song and Haidvogel (1994); theta_s = 0: uniform, C = s
        if theta_s == 0:
            return np.asarray(s, dtype=float)
        return ((1 - theta_b) * np.sinh(theta_s * s) / np.sinh(theta_s)
                + theta_b * (0.5 * np.tanh(theta_s * (s + 0.5)) / np.tanh(0.5 * theta_s) - 0.5))
    if vstretching != 4:
        raise NotImplementedError('only Vstretching = 1 and 4 are implemented')
    if theta_s > 0:
        csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    else:
        csur = -s ** 2
    if theta_b > 0:
        return (np.exp(theta_b * csur) - 1.0) / (1.0 - np.exp(-theta_b))
    return csur


def z_levels(h, zeta, p, kind='rho'):
    '''
    Depths (m, negative down) of the ROMS levels for depth array h (any shape)
    and free surface zeta, with parameters p (dict from vgrid.yaml).
    Returns an array with a leading level axis (N or N+1).
    '''
    s_w, s_r = s_levels(p['N'])
    s = s_r if kind == 'rho' else s_w
    C = stretching(s, p['theta_s'], p['theta_b'], p['Vstretching'])
    h = np.asarray(h, dtype=float)
    zeta = np.broadcast_to(zeta, h.shape)
    shape = (-1,) + (1,) * h.ndim
    if p['Vtransform'] == 2:
        z0 = (p['hc'] * s.reshape(shape) + h * C.reshape(shape)) / (p['hc'] + h)
        return zeta + (zeta + h) * z0
    z0 = p['hc'] * s.reshape(shape) + (h - p['hc']) * C.reshape(shape)
    return z0 + zeta * (1.0 + z0 / h)


def fit(KB=8, hc=0.01):
    '''Fit theta_s, theta_b of Vstretching=4 to the POM interfaces.'''
    z_pom, zz_pom, _, _ = depth(KB)
    N = KB - 1
    s_w, s_r = s_levels(N)
    target = z_pom[::-1]                 # POM runs surface->bottom, ROMS bottom->surface
    best = None
    for ts in np.arange(0.0, 10.001, 0.05):
        for tb in np.arange(0.0, 4.001, 0.05):
            err = stretching(s_w, ts, tb) - target
            rms = np.sqrt(np.mean(err ** 2))
            if best is None or rms < best[0]:
                best = (rms, ts, tb)
    rms, ts, tb = best
    p = {'N': int(N), 'Vtransform': 2, 'Vstretching': 4,
         'theta_s': float(round(ts, 2)), 'theta_b': float(round(tb, 2)),
         'hc': float(hc), 'Tcline': float(hc),
         'fit_rms_sigma': float(rms), 'pom_KB': int(KB)}
    return p, z_pom, zz_pom


def plot_levels(p, z_pom, zz_pom, out):
    s_w, s_r = s_levels(p['N'])
    cw = stretching(s_w, p['theta_s'], p['theta_b'], p['Vstretching'])
    cr = stretching(s_r, p['theta_s'], p['theta_b'], p['Vstretching'])
    fig, ax = plt.subplots(figsize=(4.5, 5))
    for z in z_pom:
        ax.axhline(z, xmin=0.05, xmax=0.45, color='#c0392b', lw=1.5)
    for z in cw:
        ax.axhline(z, xmin=0.55, xmax=0.95, color='#1f5fa8', lw=1.5)
    ax.plot(np.full(zz_pom[:-1].size, 0.25), zz_pom[:-1], 'o', color='#c0392b', ms=4)
    ax.plot(np.full(cr.size, 0.75), cr, 'o', color='#1f5fa8', ms=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 0.05)
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels([f'POM (KB={p["pom_KB"]})',
                        f'ROMS N={p["N"]}\ntheta_s={p["theta_s"]}, theta_b={p["theta_b"]}'], fontsize=8)
    ax.set_ylabel('z / h  (lines: interfaces, dots: layer centres)', fontsize=8)
    ax.set_title(f'Vertical levels (rms difference {p["fit_rms_sigma"]:.3f})', fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load():
    return yaml.safe_load(VGRID_FILE.read_text())


def main(KB=8, hc=0.01, N=DEFAULT['N'], vstretching=DEFAULT['Vstretching'], theta_s=DEFAULT['theta_s'],
         theta_b=DEFAULT['theta_b']):
    '''vstretching=None: the levels fitted to POM (N = KB-1 unless N is given).'''
    p, z_pom, zz_pom = fit(KB, hc)
    if N is not None:
        p['N'] = int(N)
    if vstretching is not None:          # explicit stretching instead of the POM fit
        p.update({'Vstretching': int(vstretching), 'theta_s': float(theta_s), 'theta_b': float(theta_b)})
        s_w, _ = s_levels(KB - 1)
        p['fit_rms_sigma'] = float(np.sqrt(np.mean((stretching(s_w, p['theta_s'], p['theta_b'], p['Vstretching'])
                                                    - z_pom[::-1]) ** 2))) if p['N'] == KB - 1 else float('nan')
        p['source'] = 'explicit (not fitted to POM)'
    s_all, _ = s_levels(p['N'])
    dC = np.diff(stretching(s_all, p['theta_s'], p['theta_b'], p['Vstretching']))
    p['min_layer_fraction'] = float(dC.min())
    p['bottom_layer_fraction'] = float(dC[0])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VGRID_FILE.write_text(
        '# ROMS vertical grid, written by ROMS/python/roms_vgrid.py\n'
        + yaml.safe_dump(p, sort_keys=False))
    plot_levels(p, z_pom, zz_pom, DATA_DIR / 'vgrid_levels.png')
    s_w, _ = s_levels(p['N'])
    cw = stretching(s_w, p['theta_s'], p['theta_b'], p['Vstretching'])
    print(f"N={p['N']} Vstretching={p['Vstretching']} theta_s={p['theta_s']} theta_b={p['theta_b']} hc={p['hc']} "
          f"rms={p['fit_rms_sigma']:.4f}")
    print('POM  z :', np.round(z_pom[::-1], 4))
    print('ROMS Cw:', np.round(cw, 4))
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROMS vertical grid (default: 10 layers, mild surface refinement)')
    parser.add_argument('--kb', type=int, default=8, help='POM KB (ROMS N = KB-1)')
    parser.add_argument('--hc', type=float, default=0.01,
                        help='critical depth hc = Tcline (m); small so levels are depth fractions')
    parser.add_argument('--n', type=int, default=DEFAULT['N'], help='number of layers')
    parser.add_argument('--vstretching', type=int, default=DEFAULT['Vstretching'], choices=[1, 4])
    parser.add_argument('--theta-s', type=float, default=DEFAULT['theta_s'])
    parser.add_argument('--theta-b', type=float, default=DEFAULT['theta_b'])
    parser.add_argument('--pom-fit', action='store_true', help='levels fitted to POM KB (the original N = 7)')
    a = parser.parse_args()
    if a.pom_fit:
        main(a.kb, a.hc, a.kb - 1, None)
    else:
        main(a.kb, a.hc, a.n, a.vstretching, a.theta_s, a.theta_b)
