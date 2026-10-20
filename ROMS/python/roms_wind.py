"""
ROMS surface wind-stress forcing file, Data/<domain>/wind.nc (FRCNAME).

The default runs are windless (ANA_SMFLUX in micro_estuary.h).
Wind runs: build with  build_roms.sh --dyes --cpp "-DUSE_WIND_FILE" --suffix wind
(micro_estuary_my25_serial_dyes_wind), write the file with --out, and run with the keys
WIND_FILE=<that file> EXEC=micro_estuary_my25_serial_dyes_wind (run_one.sh copies it as wind.nc,
the FRCNAME of the .in file).

The wind is spatially uniform and may vary in time (speed and meteorological direction
projected onto the curvilinear grid). Wind speed U (m/s at 10 m) and direction (degrees,
meteorological: the direction the wind blows FROM, 0 = from north) become a
stress with the bulk formula
    tau = rho_air * Cd(U) * U * (u_east, v_north)        (N/m2)
    Cd  = 1.2e-3 for U <= 11 m/s, (0.49 + 0.065 U) * 1e-3 above (Large and Pond 1981)
and the stress is rotated to the grid (xi, eta) directions with the grid angle and
averaged to the u and v points, as ROMS expects for sustr / svstr.

Wind input (one of):
    default            no wind: zero stress (two records, 0 and 3650 days)
    --speed --direction  constant wind
    --csv FILE         time series with columns time_days, speed_m_s, direction_deg
    --diurnal M        diurnal sea-breeze cycle along one axis: u(t) = A sin(2 pi t / 1 day), from
                       --direction (default 270, west) during the first half of each day and from the
                       opposite direction during the second, with A = M * sqrt(2) so that the median of
                       |u| is M (m/s); records every
                       15 min for DIURNAL_DAYS (40) days. With a constant drag coefficient its mean
                       |stress| equals that of a constant wind of speed M.
    --scale K          multiplies the wind SPEED (wind experiments x0.25 ... x4)

Usage: python roms_wind.py [--domains ...] [--speed 5 --direction 270 | --csv wind.csv | --diurnal 2]
                           [--scale 1] [--out wind.nc]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from common import DOMAINS, domain_dir, rho_to_u, rho_to_v
from roms_ini import TIME_UNITS

RHO_AIR = 1.22
DAY = 86400.0


def drag_coefficient(speed):
    '''Large and Pond (1981) neutral drag coefficient, constant below 11 m/s.'''
    speed = np.asarray(speed, dtype=float)
    return np.where(speed <= 11.0, 1.2e-3, (0.49 + 0.065 * speed) * 1e-3)


DIURNAL_DAYS = 40.0


def diurnal_series(median, direction=270.0):
    '''sine wind with median |u| = median, from `direction`, then from the opposite; see the docstring.'''
    t = np.arange(0.0, DIURNAL_DAYS + 1e-9, 1.0 / 96)
    u = median * np.sqrt(2.0) * np.sin(2 * np.pi * t)
    return t * DAY, np.abs(u), np.where(u >= 0, direction % 360.0, (direction + 180.0) % 360.0)


def wind_series(speed=0.0, direction=0.0, csv=None, t_end_days=3650.0, diurnal=None):
    if diurnal is not None:
        return diurnal_series(diurnal, 270.0 if direction is None else direction)
    direction = 0.0 if direction is None else direction
    if csv is None:
        return np.array([0.0, t_end_days * DAY]), np.full(2, float(speed)), np.full(2, float(direction))
    df = pd.read_csv(csv)
    return df['time_days'].to_numpy() * DAY, df['speed_m_s'].to_numpy(float), df['direction_deg'].to_numpy(float)


def stress_east_north(speed, direction):
    '''Wind stress (N/m2) eastward and northward for meteorological direction (from).'''
    to = np.radians(direction + 180.0)                      # direction the wind blows TO, from north
    u, v = speed * np.sin(to), speed * np.cos(to)
    tau = RHO_AIR * drag_coefficient(speed) * speed
    return tau * u, tau * v


def build_wind(domain, times, speed, direction, scale=1.0):
    with xr.open_dataset(domain_dir(domain) / 'grid.nc') as gr:
        angle = gr.angle.values                              # angle between xi and east (rad), (eta, xi)
    speed = scale * speed
    te, tn = stress_east_north(speed, direction)             # (t,)
    ca, sa = np.cos(angle), np.sin(angle)
    txi = te[:, None, None] * ca + tn[:, None, None] * sa     # (t, eta, xi) at rho points
    teta = -te[:, None, None] * sa + tn[:, None, None] * ca
    sustr = np.stack([rho_to_u(a) for a in txi])
    svstr = np.stack([rho_to_v(a) for a in teta])
    ts = ('sms_time',)
    ds = xr.Dataset({
        'sms_time': (ts, times, {'long_name': 'surface momentum stress time', 'units': TIME_UNITS}),
        'sustr': (('sms_time', 'eta_u', 'xi_u'), sustr,
                  {'long_name': 'surface u-momentum stress', 'units': 'newton meter-2', 'time': 'sms_time'}),
        'svstr': (('sms_time', 'eta_v', 'xi_v'), svstr,
                  {'long_name': 'surface v-momentum stress', 'units': 'newton meter-2', 'time': 'sms_time'}),
        'wind_speed': (ts, speed, {'long_name': 'wind speed (input x scale)', 'units': 'meter second-1'}),
        'wind_direction': (ts, direction, {'long_name': 'wind direction, meteorological (from), 0 = north',
                                           'units': 'degree'}),
    }, attrs={
        'type': 'ROMS FORCING file',
        'title': f'{domain} uniform wind stress',
        'grid_file': 'grid.nc',
        'source': 'ROMS/python/roms_wind.py (bulk formula, Large and Pond 1981 drag)',
        'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC',
    })
    summary = {'n_records': int(times.size), 'days': [float(times[0] / DAY), float(times[-1] / DAY)],
               'speed_m_s': [float(speed.min()), float(speed.max())], 'scale': float(scale),
               'max_stress_N_m2': float(np.hypot(te, tn).max())}
    return ds, summary


def main(domains=DOMAINS, speed=0.0, direction=0.0, csv=None, scale=1.0, out='wind.nc', diurnal=None):
    times, sp, di = wind_series(speed, direction, csv, diurnal=diurnal)
    for domain in domains:
        ds, summary = build_wind(domain, times, sp, di, scale)
        out_dir = domain_dir(domain)
        encoding = {v: {'_FillValue': None} for v in ds.data_vars}
        ds.to_netcdf(out_dir / out, format='NETCDF4_CLASSIC', encoding=encoding, unlimited_dims=['sms_time'])
        summary_file = out_dir / 'summary.json'
        existing = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        existing.setdefault('wind', {})[out] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: {out}  {summary['n_records']} records, speed {summary['speed_m_s'][0]:g}.."
              f"{summary['speed_m_s'][1]:g} m/s, max stress {summary['max_stress_N_m2']:.4f} N/m2")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ROMS uniform wind-stress forcing file')
    p.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument('--speed', type=float, default=0.0, help='constant wind speed (m/s)')
    p.add_argument('--direction', type=float, default=None,
                   help='meteorological direction (deg, from); default 0 (constant wind), 270 (--diurnal)')
    p.add_argument('--csv', default=None, help='CSV with time_days, speed_m_s, direction_deg')
    p.add_argument('--diurnal', type=float, default=None, help='diurnal sine wind with this median |u| (m/s)')
    p.add_argument('--scale', type=float, default=1.0, help='multiplies the wind speed')
    p.add_argument('--out', default='wind.nc', help='file name in Data/<domain>/ (run key WIND_FILE)')
    a = p.parse_args()
    main(a.domains, a.speed, a.direction, a.csv, a.scale, a.out, a.diurnal)
