"""
ROMS river (point source) file, Data/<domain>/rivers.nc.

Replaces the POM eastern boundary condition (bcond, "yair - river inflow"),
which imposed one uniform inflow velocity over the whole eastern face to
deliver river_inflow m3/day. Here the eastern boundary is closed and the
inflow enters through the wall face (u-points at xi = L, between the last
interior rho cell and the boundary row) as one source per wet cell:
    river_direction = 0 (u-face), negative transport = flow towards the mouth
    transport split by face cross-section area (= the POM uniform velocity)
    river_Vshape uniform over the layers
    river_salt = 0, river_temp = 22 C, river_dye_NN (with --tracers) = river values of common.PASSIVE
The file is prepared for the second calibration stage and the flow
experiments; the static run does not use it (UV_PSOURCE/TS_PSOURCE off).

Usage: python roms_rivers.py [--domains ...] [--flow 20000] [--scale 1.0] [--tracers]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from common import DOMAINS, PASSIVE, domain_dir, load_grid
from roms_ini import TIME_UNITS, TEMPERATURE

RIVER_FLOW = 20000.0      # m3/day (POMwad/params river_inflow)
DAY = 86400.0


def build_rivers(domain, flow_m3_day=RIVER_FLOW, scale=1.0, t_end_days=3650.0, tracers=False):
    g = load_grid(domain)
    h = g['h']                                     # (eta, xi)
    eta_rho, xi_rho = h.shape
    L = xi_rho - 1                                 # Fortran index of the eastern boundary rho / wall u-face
    wet = np.arange(1, eta_rho - 1)                # interior rows (Fortran and python indices coincide for eta)
    with xr.open_dataset(domain_dir(domain) / 'grid.nc') as gr:
        pn = gr.pn.values
    ini_file = domain_dir(domain) / 'ini.nc'
    if not ini_file.exists():                            # fresh input folder: only ini_raw.nc so far
        ini_file = domain_dir(domain) / 'ini_raw.nc'
    with xr.open_dataset(ini_file) as ini:
        N = ini.sizes['s_rho']

    # cross-section of each wet cell at the eastern wall face (u-point between rho L-1 and L)
    dy_face = 0.5 * (1 / pn[wet, L - 1] + 1 / pn[wet, L])
    h_face = 0.5 * (h[wet, L - 1] + h[wet, L])
    area = dy_face * h_face
    total = scale * flow_m3_day / DAY              # m3/s
    q = -total * area / area.sum()                 # negative: towards -xi (into the domain, to the mouth)

    nr = wet.size
    times = np.array([0.0, t_end_days * DAY])
    nt = times.size
    tt = ('river_time',)
    a = lambda name, units=None, time=True: {'long_name': name, **({'units': units} if units else {}),
                                             **({'time': 'river_time'} if time else {})}
    data = {
        'river': (('river',), np.arange(1, nr + 1, dtype='f8'), {'long_name': 'river runoff identification number'}),
        'river_Xposition': (('river',), np.full(nr, float(L)), {'long_name': 'river XI-position', 'LuvSrc_meaning': 'i point index of U or V face source/sink'}),
        'river_Eposition': (('river',), wet.astype('f8'), {'long_name': 'river ETA-position', 'LuvSrc_meaning': 'j point index of U or V face source/sink'}),
        'river_direction': (('river',), np.zeros(nr), {'long_name': 'river runoff direction', 'flag_values': '0, 1', 'flag_meanings': 'flow_across_U-face flow_across_V-face'}),
        'river_Vshape': (('s_rho', 'river'), np.full((N, nr), 1.0 / N), {'long_name': 'river runoff mass transport vertical profile'}),
        'river_time': (tt, times, {'long_name': 'river runoff time', 'units': TIME_UNITS}),
        'river_transport': (('river_time', 'river'), np.repeat(q[None], nt, axis=0), a('river runoff vertically integrated mass transport', 'meter3 second-1')),
        'river_temp': (('river_time', 's_rho', 'river'), np.full((nt, N, nr), TEMPERATURE), a('river runoff potential temperature', 'Celsius')),
        'river_salt': (('river_time', 's_rho', 'river'), np.zeros((nt, N, nr)), a('river runoff salinity')),
    }
    for k, (name, _, _, river, long_name, units) in enumerate(PASSIVE if tracers else (), start=1):
        data[f'river_dye_{k:02d}'] = (('river_time', 's_rho', 'river'), np.full((nt, N, nr), river),
                                      a(f'river runoff {long_name} ({name})', units))
    ds = xr.Dataset(data, attrs={
        'type': 'ROMS RIVERS file',
        'title': f'{domain} river inflow at the eastern wall ({scale * flow_m3_day:.0f} m3/day)',
        'grid_file': 'grid.nc',
        'source': 'ROMS/python/roms_rivers.py',
        'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC',
    })
    summary = {
        'flow_m3_day': float(scale * flow_m3_day), 'flow_m3_s': float(total),
        'n_sources': int(nr), 'xi_position': int(L), 'eta_positions': wet.tolist(),
        'face_area_m2': float(area.sum()), 'mean_inflow_velocity_m_s': float(total / area.sum()),
        'passive_tracers': [p[0] for p in PASSIVE] if tracers else [],
    }
    return ds, summary


def main(domains=DOMAINS, flow=RIVER_FLOW, scale=1.0, tracers=False):
    for domain in domains:
        ds, summary = build_rivers(domain, flow, scale, tracers=tracers)
        # 1 day of transport must equal the daily volume
        assert np.isclose(-ds.river_transport.values[0].sum() * DAY, scale * flow)
        out_dir = domain_dir(domain)
        encoding = {v: {'_FillValue': None} for v in ds.data_vars}
        ds.to_netcdf(out_dir / 'rivers.nc', format='NETCDF4_CLASSIC', encoding=encoding,
                     unlimited_dims=['river_time'])
        summary_file = out_dir / 'summary.json'
        existing = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        existing['rivers'] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: rivers.nc  {summary['n_sources']} sources at xi={summary['xi_position']}, "
              f"{summary['flow_m3_day']:.0f} m3/day = {summary['flow_m3_s']:.4f} m3/s, "
              f"face area {summary['face_area_m2']:.2f} m2, mean inflow {100 * summary['mean_inflow_velocity_m_s']:.2f} cm/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROMS river point-source file')
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument('--flow', type=float, default=RIVER_FLOW, help='river inflow (m3/day)')
    parser.add_argument('--scale', type=float, default=1.0, help='flow multiplier for experiments')
    parser.add_argument('--tracers', action='store_true', help='add the passive tracers (common.PASSIVE)')
    a = parser.parse_args()
    main(a.domains, a.flow, a.scale, a.tracers)
