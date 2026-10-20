"""
Default initial conditions from a short spin-up run (Data/<domain>/ini.nc).

Closed-mouth grids (Data_bar, GRIDATA/add_sea_mouth.py): the spin-up is experiments/ini_fill.cfg,
river x1 and no wind from ini_raw.nc (estuary level 0, below the crest at +0.10 m). The record at
which the outflow over the crest has reached the river flow (check_mouth.py, fill_day) is taken:
    ROMS_GRIDS=grids_bar ROMS_DATA=Data_bar python ini_from_run.py --runs-dir .../ROMS_runs/ini_fill --day <fill_day>

From the run record: zeta, u, v, ubar, vbar, temp, salt.
The passive tracers (if any) are reset to their initial values from ini_raw.nc
(river water and seawater 0, ventilation tracer 1, ages 0), and the time to 0. On the closed
mouth the sea cells keep the run's seawater tracer (dye_02), consistent with their salt.

Usage: python ini_from_run.py [--domains ...] [--runs-dir .../ROMS_runs/baseline] [--day 1]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from common import DOMAINS, REPO, domain_dir, mouth_index

FIELDS = ('zeta', 'u', 'v', 'ubar', 'vbar', 'temp', 'salt')


def from_run(domain, runs_dir, day):
    run = runs_dir / domain.lower()
    raw_file = domain_dir(domain) / 'ini_raw.nc'
    with xr.open_dataset(raw_file, decode_times=False) as raw, \
            xr.open_dataset(run / 'roms_his.nc', decode_times=False) as his:
        raw = raw.load()
        days = (his.ocean_time.values - his.ocean_time.values[0]) / 86400.0
        t = int(np.argmin(np.abs(days - day)))
        if abs(days[t] - day) > 1e-6:
            raise ValueError(f'{run}: no history record at day {day} (nearest {days[t]:g})')
        for key in ('Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'hc'):
            if not np.isclose(float(raw[key]), float(his[key])):
                raise ValueError(f'{run}: {key} differs from {raw_file} (other vertical grid?)')
        # interior points only: ROMS fills the boundary rows (outside the closed walls) from the interior
        inner = (slice(None), slice(1, -1), slice(1, -1))
        if not np.allclose(raw.salt.values[0][inner], his.salt.values[0][inner], atol=1e-4):
            raise ValueError(f'{run} did not start from {raw_file}')
        ini = raw.copy()
        for v in FIELDS:
            ini[v].values[0] = his[v].isel(ocean_time=t).values
        # closed mouth: the sea cells (seaward of the crest) keep the run's seawater tracer, so that it
        # matches their salt; from the crest inward it stays 0 and counts only new seawater
        with xr.open_dataset(run / 'grid.nc') as grid:
            m = mouth_index(grid)
        if m > 0 and 'dye_02' in his:
            ini['dye_02'].values[0][..., :m] = his.dye_02.isel(ocean_time=t).values[..., :m]
        stats = {'max_speed_m_s': float(max(np.abs(ini.u.values).max(), np.abs(ini.v.values).max())),
                 'salt': [float(ini.salt.min()), float(ini.salt.max())]}
    ini.attrs.update({
        'title': f'{domain} micro-estuary initial conditions (day {day:g} of the spin-up run {run.parent.name})',
        'source': f'ROMS/python/ini_from_run.py: {run.name} roms_his.nc at day {day:g} '
                  f'(started from ini_raw.nc, roms_ini.py); passive tracers reset',
        'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC'})
    return ini, {'source_run': str(run), 'day': day, **stats}


def main(domains=DOMAINS, runs_dir=REPO.parent / 'ROMS_runs' / 'baseline', day=1.0):
    from pathlib import Path
    runs_dir = Path(runs_dir)
    for domain in domains:
        ini, summary = from_run(domain, runs_dir, day)
        out = domain_dir(domain) / 'ini.nc'
        encoding = {v: {'_FillValue': None} for v in ini.data_vars}
        ini.to_netcdf(out, format='NETCDF4_CLASSIC', encoding=encoding, unlimited_dims=['ocean_time'])
        summary_file = domain_dir(domain) / 'summary.json'
        existing = json.loads(summary_file.read_text())
        existing['ini_from_run'] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: ini.nc = day {day:g} of {runs_dir.name}/{domain.lower()}, "
              f"max speed {summary['max_speed_m_s']:.3f} m/s, salt {summary['salt'][0]:.2f}-{summary['salt'][1]:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Default initial conditions from day 1 of a static run')
    p.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument('--runs-dir', default=str(REPO.parent / 'ROMS_runs' / 'baseline'))
    p.add_argument('--day', type=float, default=1.0)
    a = p.parse_args()
    main(a.domains, a.runs_dir, a.day)
