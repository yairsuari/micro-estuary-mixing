"""
ROMS boundary file (Data/<domain>/bry.nc) for the western (sea) boundary.
The other boundaries are closed and need no data.

Sea state imposed at the mouth (as in POM wadseamount):
    zeta_west = sea level (default 0 m). An optional time series (CSV with
                columns time_days, zeta_m) can be given for sea-level / surge
                experiments. It is floored at Dcrit - h, as in the initial
                conditions, so the dry bar column stays consistent with WET_DRY.
    ubar, vbar, u, v = 0
    temp_west = 22 C
    salt_west = 39.5 (POM sbw: Mediterranean sea water), or with --salt-profile ic
                the initial-condition salinity of the mouth column (TS_A4 profile).
    dye_west_NN (with --tracers) = sea values of common.PASSIVE (01..04).

Without a time series, two records (t = 0 and t = 3650 days) hold the
constant values, so any run length is covered.

Usage: python roms_bry.py [--domains ...] [--sea-salt 39.5 | --salt-profile ic]
                          [--zeta-csv file] [--tracers]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from common import DOMAINS, PASSIVE, domain_dir, load_grid
from roms_ini import TIME_UNITS, TEMPERATURE

SEA_SALT = 39.5          # POMwad/pom08.f wadseamount: sbw(j,k) = 39.5
DAY = 86400.0


def sea_level_series(zeta_csv=None, constant=0.0, t_end_days=3650.0):
    '''Times (s) and sea level (m) at the mouth.'''
    if zeta_csv is None:
        return np.array([0.0, t_end_days * DAY]), np.array([constant, constant])
    df = pd.read_csv(zeta_csv)
    return df['time_days'].to_numpy() * DAY, df['zeta_m'].to_numpy()


def build_bry(domain, times, zeta_mouth, salt_profile='sea', sea_salt=SEA_SALT,
              dcrit=0.05, tracers=False):
    g = load_grid(domain)
    h_w = g['h'][:, 0]                                  # western boundary rho column (eta)
    eta_rho = h_w.size
    nt = times.size
    ini_file = domain_dir(domain) / 'ini.nc'
    if not ini_file.exists():                            # fresh input folder: only ini_raw.nc so far
        ini_file = domain_dir(domain) / 'ini_raw.nc'
    with xr.open_dataset(ini_file) as ini:
        N = ini.sizes['s_rho']
        salt_ic = ini.salt.values[0, :, :, 0]           # (N, eta) mouth column

    zeta = np.maximum(zeta_mouth[:, None], dcrit - h_w[None, :])        # (t, eta)
    if salt_profile == 'sea':
        salt = np.full((nt, N, eta_rho), sea_salt)
    elif salt_profile == 'ic':
        salt = np.repeat(salt_ic[None], nt, axis=0)
    else:
        raise ValueError("salt_profile must be 'sea' or 'ic'")
    temp = np.full((nt, N, eta_rho), TEMPERATURE)

    tb = ('bry_time',)
    attrs = lambda name, units=None: {'long_name': name, 'time': 'bry_time', **({'units': units} if units else {})}
    data = {
        'bry_time': (tb, times, {'long_name': 'open boundary conditions time', 'units': TIME_UNITS}),
        'zeta_west': (('bry_time', 'eta_rho'), zeta, attrs('free-surface western boundary condition', 'meter')),
        'ubar_west': (('bry_time', 'eta_u'), np.zeros((nt, eta_rho)),
                      attrs('2D u-momentum western boundary condition', 'meter second-1')),
        'vbar_west': (('bry_time', 'eta_v'), np.zeros((nt, eta_rho - 1)),
                      attrs('2D v-momentum western boundary condition', 'meter second-1')),
        'u_west': (('bry_time', 's_rho', 'eta_u'), np.zeros((nt, N, eta_rho)),
                   attrs('3D u-momentum western boundary condition', 'meter second-1')),
        'v_west': (('bry_time', 's_rho', 'eta_v'), np.zeros((nt, N, eta_rho - 1)),
                   attrs('3D v-momentum western boundary condition', 'meter second-1')),
        'temp_west': (('bry_time', 's_rho', 'eta_rho'), temp,
                      attrs('potential temperature western boundary condition', 'Celsius')),
        'salt_west': (('bry_time', 's_rho', 'eta_rho'), salt, attrs('salinity western boundary condition')),
    }
    for k, (name, _, sea, _, long_name, units) in enumerate(PASSIVE if tracers else (), start=1):
        data[f'dye_west_{k:02d}'] = (('bry_time', 's_rho', 'eta_rho'), np.full((nt, N, eta_rho), sea),
                                     attrs(f'{long_name} ({name}) western boundary condition', units))
    ds = xr.Dataset(data, attrs={
        'type': 'ROMS BOUNDARY file',
        'title': f'{domain} micro-estuary western (sea) boundary',
        'grid_file': 'grid.nc',
        'source': 'ROMS/python/roms_bry.py',
        'salt_west': f'{salt_profile} ({sea_salt} uniform)' if salt_profile == 'sea' else 'initial mouth column',
        'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC',
    })
    summary = {
        'times_days': [float(times[0] / DAY), float(times[-1] / DAY)], 'n_records': int(nt),
        'zeta_west_m': [float(zeta.min()), float(zeta.max())],
        'salt_west': [float(salt.min()), float(salt.max())], 'salt_profile': salt_profile,
        # the mouth (bar or crest) row, which is the boundary column only for the old grids
        'mouth_h_m': [float(g['h'][1:-1, g['mouth_index']].min()), float(g['h'][1:-1, g['mouth_index']].max())],
        'mouth_dry_at_zeta0': bool(np.all(g['h'][1:-1, g['mouth_index']] < dcrit)),
        'boundary_h_m': [float(h_w[1:-1].min()), float(h_w[1:-1].max())],
        'passive_tracers': [p[0] for p in PASSIVE] if tracers else [],
    }
    return ds, summary


def main(domains=DOMAINS, salt_profile='sea', sea_salt=SEA_SALT, zeta_csv=None, dcrit=0.05,
         tracers=False):
    times, zeta_mouth = sea_level_series(zeta_csv)
    for domain in domains:
        ds, summary = build_bry(domain, times, zeta_mouth, salt_profile, sea_salt, dcrit, tracers)
        out_dir = domain_dir(domain)
        encoding = {v: {'_FillValue': None} for v in ds.data_vars}
        ds.to_netcdf(out_dir / 'bry.nc', format='NETCDF4_CLASSIC', encoding=encoding,
                     unlimited_dims=['bry_time'])
        summary_file = out_dir / 'summary.json'
        existing = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        existing['bry'] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: bry.nc  {summary['n_records']} records over {summary['times_days'][1]:.0f} days, "
              f"zeta_west {summary['zeta_west_m'][0]:.3f}..{summary['zeta_west_m'][1]:.3f} m, "
              f"salt_west {summary['salt_west'][0]:.2f}..{summary['salt_west'][1]:.2f}, "
              f"mouth h {summary['mouth_h_m'][0]:.2f} m (dry at zeta=0: {summary['mouth_dry_at_zeta0']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROMS western (sea) boundary file')
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument('--salt-profile', choices=['sea', 'ic'], default='sea',
                        help="'sea': uniform --sea-salt (POM sbw); 'ic': initial mouth column")
    parser.add_argument('--sea-salt', type=float, default=SEA_SALT)
    parser.add_argument('--zeta-csv', default=None, help='CSV with time_days, zeta_m (sea level at the mouth)')
    parser.add_argument('--dcrit', type=float, default=0.05)
    parser.add_argument('--tracers', action='store_true', help='add the passive tracers (common.PASSIVE)')
    a = parser.parse_args()
    main(a.domains, a.salt_profile, a.sea_salt, a.zeta_csv, a.dcrit, a.tracers)
