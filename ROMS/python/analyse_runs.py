"""
Sanity checks and summary of the runs of one experiment (any ROMS_runs/<experiment>).

For every run folder with a roms_his.nc it computes, at each history record:
    volume, salt content, interior mean bottom-surface salinity difference,
    kinetic energy, NaN count, river water and ventilation age (if tracers),
    salinity 0.2 m below the surface and 0.2 m above the bed at the stations
    A4 (~400 m from the mouth) and A3 (~3.5 km), centreline cells,
and between records the effective vertical diffusivity of the whole estuary,
    K_eff = -(1/2) d/dt (salt variance) / integral (dS/dz)^2 dV,
i.e. all the vertical mixing the model does (turbulence closure, background and the
numerical mixing of advection). With the river off and the mouth closed this is exact;
with the river on the outflow also removes variance, so K_eff is an upper bound.

Writes in ROMS_runs/<experiment>/output_figures/:
    summary.csv           one row per run (end values, conservation, K_eff, CPU cost)
    timeseries.csv        all runs, all records
    timeseries.png        mean dS, salt, volume, kinetic energy, K_eff
    stations.png          A4 and A3 surface / bottom salinity

Usage: python analyse_runs.py EXPERIMENT [--runs-dir ROMS_runs]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import REPO, estuary_slice, mouth_index
from compare_layers import depths

RUNS_DIR = REPO.parent / 'ROMS_runs'
STATIONS = {'A4': 400.0, 'A3': 3500.0}      # distance from the mouth along the centreline (m)
SENSOR_OFFSET = 0.2                         # m below the surface / above the bed
COLOURS = ['#1f5fa8', '#e08a1e', '#2a9d4b', '#b2392f', '#7a4fb0', '#5c5c5c']


def cpu_seconds(path):
    '''CPU time ROMS reports ("Total:"); wall time also counts the time WSL was suspended.'''
    for line in (path / 'roms.log').read_text(errors='replace').splitlines():
        if line.strip().startswith('Total:'):
            return float(line.split()[1])
    return np.nan


def station_cells(grid):
    '''centre row and the xi index of each station, by distance along the centreline from the mouth
    (the bar or crest row, mouth_index; row 0 for the old grids)'''
    j = grid.sizes['eta_rho'] // 2
    m = mouth_index(grid)
    x, y = grid.x_rho.values[j, m:], grid.y_rho.values[j, m:]
    dist = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return j, {name: m + int(np.argmin(np.abs(dist - d))) for name, d in STATIONS.items()}


def sensor_values(s, zr, zw, j, i):
    '''Salinity at SENSOR_OFFSET below the surface and above the bed of column (j, i).'''
    z, col = zr[:, j, i], s[:, j, i]
    top, bed = zw[-1, j, i], zw[0, j, i]
    return (float(np.interp(top - SENSOR_OFFSET, z, col)), float(np.interp(bed + SENSOR_OFFSET, z, col)))


def run_series(path):
    ds = xr.open_dataset(path / 'roms_his.nc', decode_times=False)
    grid = xr.open_dataset(path / 'grid.nc')
    dA = 1.0 / (grid.pm.values * grid.pn.values)
    j, cells = station_cells(grid)
    inner = estuary_slice(grid, 3)                  # estuary cells (no sea rows, mouth or walls)
    m = mouth_index(grid)
    tracers = 'dye_01' in ds
    if tracers:
        last = max(int(v[4:6]) for v in ds.data_vars if v.startswith('dye_') and v[4:6].isdigit())
    rows = []
    for t in range(ds.sizes['ocean_time']):
        zw = depths(ds, t)
        zr = depths(ds, t, 'r')
        dz = np.diff(zw, axis=0)
        dV = dz * dA
        s = ds.salt.isel(ocean_time=t).values
        si, dVi = s[inner], dV[inner]
        mean = (si * dVi).sum() / dVi.sum()
        dsdz = np.diff(s, axis=0) / np.diff(zr, axis=0)                   # interior w levels
        dzw = np.diff(zr, axis=0)
        u = ds.u.isel(ocean_time=t).values
        v = ds.v.isel(ocean_time=t).values
        row = {'day': float(ds.ocean_time[t] - ds.ocean_time[0]) / 86400.0,
               'volume_m3': dVi.sum(), 'salt_content': (si * dVi).sum(),
               'salt_variance': ((si - mean) ** 2 * dVi).sum(),
               'grad2': ((dsdz ** 2) * dzw * dA)[inner].sum(),
               'mean_dS': float(np.mean(si[0] - si[-1])),
               'kinetic_energy': 0.5 * (np.mean(u ** 2) + np.mean(v ** 2)),
               'max_speed_interior': float(max(np.abs(u[:, 1:-1, m + 1:-1]).max(), np.abs(v[:, 1:-1, m + 1:-1]).max())),
               'nan': int(np.isnan(s).sum() + np.isnan(u).sum())}
        for name, i in cells.items():
            row[f'{name}_surface'], row[f'{name}_bottom'] = sensor_values(s, zr, zw, j, i)
        if tracers:
            c1 = ds.dye_01.isel(ocean_time=t).values[inner]
            c2 = ds[f'dye_{last:02d}'].isel(ocean_time=t).values[inner]
            a2 = ds[f'dye_{last:02d}_age'].isel(ocean_time=t).values[inner]
            row['river_water_m3'] = (c1 * dVi).sum()
            row['ventilation_age_h'] = (a2 * dVi).sum() / max((c2 * dVi).sum(), 1e-12) / 3600.0
        rows.append(row)
    df = pd.DataFrame(rows)
    dt = np.diff(df.day.values) * 86400.0
    keff = -0.5 * np.diff(df.salt_variance.values) / dt / (0.5 * (df.grad2.values[1:] + df.grad2.values[:-1]))
    df['K_eff'] = np.concatenate(([np.nan], keff))
    return df


def k_eff_between(df, d0, d1):
    a, b = (int(np.argmin(np.abs(df.day.values - d))) for d in (d0, d1))
    if b <= a:
        return np.nan
    return (-0.5 * (df.salt_variance.iloc[b] - df.salt_variance.iloc[a]) / ((df.day.iloc[b] - df.day.iloc[a]) * 86400.0)
            / df.grad2.iloc[a:b + 1].mean())


def timeseries_figure(series, out):
    metrics = [('mean_dS', 'bottom − surface salinity\n(interior mean)', False),
               ('salt_content', 'salt content\n(relative to start)', False),
               ('volume_m3', 'volume\n(relative to start)', False),
               ('kinetic_energy', 'kinetic energy\n(m² s⁻², mean)', True),
               ('K_eff', 'effective vertical\ndiffusivity (m² s⁻¹)', True)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.0 * len(metrics)), sharex=True)
    for n, (run, df) in enumerate(series.items()):
        for ax, (key, label, log) in zip(axes, metrics):
            y = df[key] / df[key].iloc[0] if key in ('salt_content', 'volume_m3') else df[key]
            ax.plot(df.day, y, color=COLOURS[n % len(COLOURS)], lw=1.5, label=run)
    for ax, (key, label, log) in zip(axes, metrics):
        ax.set_ylabel(label, fontsize=8)
        if log:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8, ncol=len(series))
    axes[-1].set_xlabel('days')
    fig.tight_layout()
    fig.savefig(out / 'timeseries.png', dpi=120)
    plt.close(fig)


def stations_figure(series, out):
    fig, axes = plt.subplots(1, len(STATIONS), figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, STATIONS):
        for n, (run, df) in enumerate(series.items()):
            c = COLOURS[n % len(COLOURS)]
            ax.plot(df.day, df[f'{name}_bottom'], color=c, lw=1.6, label=f'{run} bottom')
            ax.plot(df.day, df[f'{name}_surface'], color=c, lw=1.0, ls='--', label=f'{run} surface')
        ax.set_title(f'{name} ({STATIONS[name] / 1000:g} km from the mouth), 0.2 m from surface / bed',
                     fontsize=9, loc='left')
        ax.set_xlabel('days')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('salinity')
    axes[-1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'stations.png', dpi=120)
    plt.close(fig)


def main(experiment, runs_dir=RUNS_DIR):
    exp = Path(runs_dir) / experiment
    out = exp / 'output_figures'
    out.mkdir(exist_ok=True)
    series, rows = {}, []
    for p in sorted(exp.iterdir()):
        if not (p / 'roms_his.nc').exists():
            continue
        df = run_series(p)
        series[p.name] = df
        meta = json.loads((p / 'run_meta.json').read_text())
        cpu = cpu_seconds(p)
        end = df.iloc[-1]
        rows.append({'run': p.name, 'exec': meta['parameters'].get('EXEC'), 'days': end.day,
                     'cpu_min_per_sim_day': cpu / 60.0 / end.day if end.day else np.nan,
                     'nan': int(df.nan.sum()),
                     'volume_change_pct': 100 * (end.volume_m3 / df.volume_m3.iloc[0] - 1),
                     'salt_change_pct': 100 * (end.salt_content / df.salt_content.iloc[0] - 1),
                     'mean_dS_start': df.mean_dS.iloc[0], 'mean_dS_day1': df.mean_dS.iloc[int(np.argmin(np.abs(df.day - 1)))],
                     'mean_dS_end': end.mean_dS,
                     'K_eff_day0_1': k_eff_between(df, 0, 1), 'K_eff_day2_end': k_eff_between(df, 2, end.day),
                     'max_speed_interior_end': end.max_speed_interior,
                     'A4_bottom_start': df.A4_bottom.iloc[0], 'A4_bottom_end': end.A4_bottom,
                     'A3_bottom_start': df.A3_bottom.iloc[0], 'A3_bottom_end': end.A3_bottom,
                     'river_water_m3_end': end.get('river_water_m3', np.nan),
                     'ventilation_age_h_end': end.get('ventilation_age_h', np.nan)})
        df.insert(0, 'run', p.name)
    if not series:
        raise SystemExit(f'no runs with roms_his.nc in {exp}')
    table = pd.DataFrame(rows)
    table.to_csv(out / 'summary.csv', index=False)
    pd.concat(series.values()).to_csv(out / 'timeseries.csv', index=False)
    for df in series.values():
        df.drop(columns='run', inplace=True)
    timeseries_figure(series, out)
    stations_figure(series, out)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(table.T.to_string(header=False))
    print(f'written to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Sanity checks and summary of the runs of one experiment')
    p.add_argument('experiment')
    p.add_argument('--runs-dir', default=str(RUNS_DIR))
    a = p.parse_args()
    main(a.experiment, a.runs_dir)
