"""
Checks of the closed mouth (sea cells + raised crest, GRIDATA/add_sea_mouth.py) for a set of runs.

Per hourly record, over the estuary cells (common.estuary_slice):
    level_m       mean free surface of the estuary
    volume_m3     water volume
    salt          salt content (PSU m3)
    seawater_m3   volume of the seawater tracer (dye_02)
    seawater_max  highest seawater-tracer concentration in any estuary cell (and in the first cell
                  behind the crest), so that a leak shows even when its volume is small
    outflow_m3_s  outflow over the crest = river inflow - d(volume)/dt
    crest_wet     fraction of the crest cells with zeta + h > Dcrit
and, per run, the first day at which the outflow is within --tol of the river flow (the end of
the filling), the seawater entry rate and the drift of salt and volume.

Usage: python check_mouth.py RUN_DIR [RUN_DIR ...] [--tol 0.02] [--dcrit 0.05]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import estuary_slice, mouth_index
from compare_layers import depths


def river_flow(run):
    '''river inflow (m3/s) of the run: the copied rivers.nc already holds RIVER_SCALE (run_one.sh)'''
    params = json.loads((run / 'run_meta.json').read_text()).get('parameters', {})
    if params.get('RIVERS', 'off') != 'on' or not (run / 'rivers.nc').exists():
        return 0.0
    with xr.open_dataset(run / 'rivers.nc', decode_times=False) as r:
        return float(-r.river_transport.isel(river_time=0).sum())


def series(run, dcrit=0.05):
    ds = xr.open_dataset(run / 'roms_his.nc', decode_times=False)
    grid = xr.open_dataset(run / 'grid.nc')
    inner = estuary_slice(grid)
    m = mouth_index(grid)
    dA = 1.0 / (grid.pm.values * grid.pn.values)
    h = grid.h.values
    rows = []
    for t in range(ds.sizes['ocean_time']):
        zeta = ds.zeta.isel(ocean_time=t).values
        zw = depths(ds, t)
        dV = np.diff(zw, axis=0) * dA
        s = ds.salt.isel(ocean_time=t).values
        row = {'day': float(ds.ocean_time[t] - ds.ocean_time[0]) / 86400.0,
               'level_m': float(np.average(zeta[inner], weights=dA[inner])),
               'volume_m3': float(dV[(slice(None),) + inner].sum()),
               'salt': float((s * dV)[(slice(None),) + inner].sum()),
               'crest_wet': float(np.mean((zeta + h)[1:-1, m] > dcrit + 1e-6)),
               'nan': int(np.isnan(s).sum())}
        if 'dye_02' in ds:
            c2 = ds.dye_02.isel(ocean_time=t).values
            row['seawater_m3'] = float((c2 * dV)[(slice(None),) + inner].sum())
            row['seawater_max'] = float(c2[(slice(None),) + inner].max())         # any estuary cell
            row['seawater_max_inner_cell'] = float(c2[:, 1:-1, m + 1].max())     # first cell behind the crest
        rows.append(row)
    df = pd.DataFrame(rows)
    q_river = river_flow(run)
    dvdt = np.gradient(df.volume_m3.values, df.day.values * 86400.0)
    df['outflow_m3_s'] = q_river - dvdt
    return df, q_river


def summary(run, df, q_river, tol):
    out = {'run': run.name, 'river_m3_s': q_river, 'days': df.day.iloc[-1],
           'level_m': [round(df.level_m.min(), 4), round(df.level_m.iloc[-1], 4)],
           'crest_wet_max': df.crest_wet.max(), 'nan': int(df.nan.sum())}
    if q_river > 0:
        ok = df[(df.day > 0.1) & (np.abs(df.outflow_m3_s - q_river) <= tol * q_river)]
        out['fill_day'] = float(ok.day.iloc[0]) if len(ok) else None
        late = df[df.day > df.day.iloc[-1] - 1.0]         # a whole day, so that a daily wind cycle averages out
        out['outflow_late_over_river'] = round(float(late.outflow_m3_s.mean() / q_river), 4)
    else:
        out['volume_drift'] = float(df.volume_m3.iloc[-1] / df.volume_m3.iloc[0] - 1)
    out['salt_drift'] = float(df.salt.iloc[-1] / df.salt.iloc[0] - 1)
    if 'seawater_m3' in df:
        out['seawater_m3_end'] = round(float(df.seawater_m3.iloc[-1]), 3)
        out['seawater_m3_per_day'] = round(float(np.polyfit(df.day, df.seawater_m3, 1)[0]), 3)
        out['seawater_max'] = float(df.seawater_max.max())
        out['seawater_max_inner_cell'] = float(df.seawater_max_inner_cell.max())
    return out


def main():
    p = argparse.ArgumentParser(description='Checks of the closed mouth')
    p.add_argument('runs', nargs='+')
    p.add_argument('--tol', type=float, default=0.02)
    p.add_argument('--dcrit', type=float, default=0.05)
    a = p.parse_args()
    rows = []
    for r in a.runs:
        run = Path(r)
        df, q = series(run, a.dcrit)
        df.to_csv(run / 'check_mouth.csv', index=False, float_format='%.6g')
        rows.append(summary(run, df, q, a.tol))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
