"""
ROMS prescribed initial conditions (Data/<domain>/ini_raw.nc), the same state as the POM IC:
    zeta = 0, except on cells shallower than Dcrit, where zeta = Dcrit - h
           (the minimum water column ROMS WET_DRY expects on dry cells, e.g. the mouth bar)
    u = v = ubar = vbar = 0
    temp = 22 C
    salt from get_S in GRIDATA/grid_IC.py (the TS_A4 profile at the mouth,
         interpolated linearly along the centreline to 0 at the river end),
         evaluated at the depth of each ROMS level below the surface; with the closed-mouth
         grids (ROMS_GRIDS=grids_bar) the profile starts at the crest (mouth_index) and the
         sea rows seaward of it hold SEA_SALT
With --tracers, the passive tracers of common.PASSIVE (river water fraction
and ventilation age, two mean-age pairs) at their initial values.

Needs Data/vgrid.yaml (roms_vgrid.py) and the grid npz files.

Runs start by default from ini.nc, the state of a static run from ini_raw.nc at day 1
(ini_from_run.py), so they need no spin-up; run key INI_FILE=ini_raw.nc starts from these
prescribed fields instead.

Usage: python roms_ini.py [--domains ...] [--dcrit 0.05] [--tracers]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import DOMAINS, GRIDATA, PASSIVE, domain_dir, estuary_slice, load_grid
from roms_vgrid import load as load_vgrid, s_levels, stretching, z_levels

sys.path.insert(0, str(GRIDATA))
from grid_IC import get_S            # noqa: E402

TIME_UNITS = 'seconds since 2000-01-01 00:00:00'     # ROMS TIME_REF = 20000101.0d0
TEMPERATURE = 22.0
SEA_SALT = 39.5          # sea rows of the closed-mouth grids (as roms_bry.SEA_SALT)
BOTTOM = '#8b5a2b'


def build_ini(domain, dcrit=0.05, tracers=False):
    g = load_grid(domain)
    p = load_vgrid()
    h = g['h']                                       # (eta, xi)
    eta_rho, xi_rho = h.shape
    N = p['N']

    zeta = np.where(h < dcrit, dcrit - h, 0.0)
    z_r = z_levels(h, zeta, p, 'rho')                # (N, eta, xi), bottom first
    z_w = z_levels(h, zeta, p, 'w')                  # (N+1, eta, xi)

    # get_S works in (i, j, k) = (xi, eta, level) and takes depth fractions
    # relative to h; use the depth of each level below the free surface
    # (the estuary part only, from the mouth row, so that the profile starts at the mouth)
    m = g['mouth_index']
    frac = np.transpose((z_r - zeta) / h, (2, 1, 0))[m:]
    x_ij, y_ij, h_ij = g['x'].T[m:], g['y'].T[m:], h.T[m:]
    _, S_ijk = get_S(x_ij, y_ij, xi_rho - m, eta_rho, N, h_ij, frac)
    salt = np.full((N, eta_rho, xi_rho), SEA_SALT)
    salt[:, :, m:] = np.transpose(S_ijk, (2, 1, 0))  # (N, eta, xi)
    temp = np.full_like(salt, TEMPERATURE)

    assert np.all(np.isfinite(salt)) and salt.min() >= 0
    s_w, s_r = s_levels(N)
    Cs_w = stretching(s_w, p['theta_s'], p['theta_b'], p['Vstretching'])
    Cs_r = stretching(s_r, p['theta_s'], p['theta_b'], p['Vstretching'])

    t = ('ocean_time',)
    rho = ('ocean_time', 'eta_rho', 'xi_rho')
    r3 = ('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
    u3 = ('ocean_time', 's_rho', 'eta_u', 'xi_u')
    v3 = ('ocean_time', 's_rho', 'eta_v', 'xi_v')
    zeros = lambda *shape: np.zeros((1,) + shape)

    data = {
        'Vtransform': ((), np.int32(p['Vtransform']), {'long_name': 'vertical terrain-following transformation equation'}),
        'Vstretching': ((), np.int32(p['Vstretching']), {'long_name': 'vertical terrain-following stretching function'}),
        'theta_s': ((), np.float64(p['theta_s']), {'long_name': 'S-coordinate surface control parameter'}),
        'theta_b': ((), np.float64(p['theta_b']), {'long_name': 'S-coordinate bottom control parameter'}),
        'Tcline': ((), np.float64(p['Tcline']), {'long_name': 'S-coordinate surface/bottom layer width', 'units': 'meter'}),
        'hc': ((), np.float64(p['hc']), {'long_name': 'S-coordinate parameter, critical depth', 'units': 'meter'}),
        's_rho': (('s_rho',), s_r, {'long_name': 'S-coordinate at RHO-points', 'valid_min': -1.0, 'valid_max': 0.0}),
        's_w': (('s_w',), s_w, {'long_name': 'S-coordinate at W-points', 'valid_min': -1.0, 'valid_max': 0.0}),
        'Cs_r': (('s_rho',), Cs_r, {'long_name': 'S-coordinate stretching curves at RHO-points'}),
        'Cs_w': (('s_w',), Cs_w, {'long_name': 'S-coordinate stretching curves at W-points'}),
        'ocean_time': (t, np.array([0.0]), {'long_name': 'time since initialization', 'units': TIME_UNITS}),
        'zeta': (rho, zeta[None], {'long_name': 'free-surface', 'units': 'meter', 'time': 'ocean_time'}),
        'ubar': (('ocean_time', 'eta_u', 'xi_u'), zeros(eta_rho, xi_rho - 1),
                 {'long_name': 'vertically integrated u-momentum component', 'units': 'meter second-1', 'time': 'ocean_time'}),
        'vbar': (('ocean_time', 'eta_v', 'xi_v'), zeros(eta_rho - 1, xi_rho),
                 {'long_name': 'vertically integrated v-momentum component', 'units': 'meter second-1', 'time': 'ocean_time'}),
        'u': (u3, zeros(N, eta_rho, xi_rho - 1), {'long_name': 'u-momentum component', 'units': 'meter second-1', 'time': 'ocean_time'}),
        'v': (v3, zeros(N, eta_rho - 1, xi_rho), {'long_name': 'v-momentum component', 'units': 'meter second-1', 'time': 'ocean_time'}),
        'temp': (r3, temp[None], {'long_name': 'potential temperature', 'units': 'Celsius', 'time': 'ocean_time'}),
        'salt': (r3, salt[None], {'long_name': 'salinity', 'time': 'ocean_time'}),
    }
    for name, initial, _, _, long_name, units in (PASSIVE if tracers else ()):
        data[name] = (r3, np.full((1, N, eta_rho, xi_rho), initial),
                      {'long_name': long_name, 'units': units, 'time': 'ocean_time'})
    ds = xr.Dataset(data, attrs={
        'type': 'ROMS INITIAL file',
        'title': f'{domain} micro-estuary initial conditions',
        'grid_file': 'grid.nc',
        'source': 'ROMS/python/roms_ini.py (salinity from GRIDATA/grid_IC.py get_S)',
        'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC',
    })

    # totals over the estuary cells, as a reference for conservation checks
    inner = estuary_slice(g)
    with xr.open_dataset(domain_dir(domain) / 'grid.nc') as gr:
        dA = 1.0 / (gr.pm.values[inner] * gr.pn.values[inner])
    Hz = np.diff(z_w, axis=0)[(slice(None),) + inner]
    vol = np.sum(Hz * dA)
    summary = {
        'zeta_m': [float(zeta.min()), float(zeta.max())],
        'cells_with_raised_zeta': int(np.sum(zeta > 0)),
        'salt': [float(salt.min()), float(salt.max())],
        'interior_volume_m3': float(vol),
        'interior_mean_salinity': float(np.sum(salt[(slice(None),) + inner] * Hz * dA) / vol),
        'passive_tracers': [p[0] for p in PASSIVE] if tracers else [],
        'time_units': TIME_UNITS,
    }
    return ds, summary, z_r


def long_section(sections, out):
    '''Initial salinity along the centreline on the ROMS levels, bottom in brown.'''
    hmax = max(s['h'].max() for s in sections.values())
    smax = max(s['salt'].max() for s in sections.values())
    levels = np.linspace(0, np.ceil(smax), 16)
    fig, axes = plt.subplots(len(sections), 1, figsize=(10, 2.5 * len(sections)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, sections.items()):
        # extend the bottom/top level values to the bed and the surface for the fill
        z = np.vstack([-s['h'], s['z'], np.zeros_like(s['h'])])
        sal = np.vstack([s['salt'][:1], s['salt'], s['salt'][-1:]])
        dist = np.repeat(s['dist'][None, :], z.shape[0], axis=0)
        cf = ax.contourf(dist, z, sal, levels=levels, cmap='Blues')
        ax.plot(dist[1:-1].T, s['z'].T, color='white', lw=0.3, alpha=0.6)   # ROMS level centres
        ax.fill_between(s['dist'], -s['h'], -hmax * 1.15, color=BOTTOM, lw=0)
        ax.plot(s['dist'], -s['h'], color='#5b3a1a', lw=1)
        ax.set_ylim(-hmax * 1.15, 0.1)
        ax.set_xlim(0, s['dist'][-1])
        ax.set_title(f'{name} (ROMS levels, centreline)', fontsize=10, loc='left', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel('Distance from the mouth along the centreline (km)', fontsize=8)
    cb = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label('Salinity', fontsize=8)
    fig.suptitle('ROMS initial salinity (white lines: level centres)', x=0.06, ha='left', fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(domains=DOMAINS, dcrit=0.05, tracers=False, figure=None):
    sections = {}
    for domain in domains:
        ds, summary, z_r = build_ini(domain, dcrit, tracers)
        out_dir = domain_dir(domain)
        encoding = {v: {'_FillValue': None} for v in ds.data_vars}
        ds.to_netcdf(out_dir / 'ini_raw.nc', format='NETCDF4_CLASSIC', encoding=encoding,
                     unlimited_dims=['ocean_time'])
        summary_file = out_dir / 'summary.json'
        existing = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        existing['ini'] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: ini_raw.nc  salt {summary['salt'][0]:.2f}-{summary['salt'][1]:.2f}, "
              f"zeta raised on {summary['cells_with_raised_zeta']} cells (max {summary['zeta_m'][1]:.3f} m), "
              f"interior volume {summary['interior_volume_m3']:.0f} m3, "
              f"mean salinity {summary['interior_mean_salinity']:.2f}")
        g = load_grid(domain)
        c = g['h'].shape[0] // 2
        x, y = g['x'][c], g['y'][c]
        sections[domain] = {
            'dist': np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / 1e3,
            'z': z_r[:, c, :], 'salt': ds.salt.values[0, :, c, :], 'h': g['h'][c],
        }
    fig_path = figure or (domain_dir(domains[0]).parent / 'ini_salinity_section.png')
    long_section(sections, fig_path)
    return sections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROMS initial conditions')
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument('--dcrit', type=float, default=0.05, help='WET_DRY critical depth (m)')
    parser.add_argument('--tracers', action='store_true', help='add the passive tracers (common.PASSIVE)')
    parser.add_argument('--figure', default=None, help='output path of the salinity section figure')
    a = parser.parse_args()
    main(a.domains, a.dcrit, a.tracers, a.figure)
