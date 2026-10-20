"""
Compare runs with different numbers of vertical layers; also shared functions (depths of the model levels).

Runs are named <layers>_<case>, e.g. N20u_river4: layers N07 (POM-fitted levels),
N10u, N20u (uniform); cases static, river1, river4.

For each case it writes, in ROMS_runs/vertical_layers/output_figures/:
    section_<case>.mp4     clip of the along-channel salinity section on the centreline
                           (eta = JM//2), one panel per layer set, hourly frames
    frames_<case>.png      the same section at 0, 6, 12, 24, 48 and 72 h (rows) for each
                           layer set (columns)
and for all runs together:
    timeseries.png         interior mean bottom-surface salinity difference, salt content,
                           volume, river water in the estuary, mean ventilation age
    summary.csv            numbers at the end of each run and the cost (wall time)

Usage: python compare_layers.py [--runs-dir .../ROMS_runs/vertical_layers] [--no-video]
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
from matplotlib import animation

from common import REPO

try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

LAYERS = {'N07': 'N = 7 (POM levels)', 'N10u': 'N = 10 uniform', 'N20u': 'N = 20 uniform'}
CASES = {'static': 'no river', 'river1': 'river 20 000 m³/day', 'river4': 'river 80 000 m³/day'}
COLOURS = {'N07': '#1f5fa8', 'N10u': '#e08a1e', 'N20u': '#2a9d4b'}
BOTTOM = '#8b5a2b'
FRAME_HOURS = (0, 6, 12, 24, 48, 72)
SALT_LEVELS = np.arange(0, 30.1, 1.0)
OUT = REPO.parent / 'ROMS_runs' / 'vertical_layers' / 'output_figures'     # set from --runs-dir in main


def depths(ds, t, where='w'):
    '''z of the w (interfaces) or rho levels, (N[+1], eta, xi), any Vtransform 2 file.'''
    h = ds.h.values
    zeta = ds.zeta.isel(ocean_time=t).values
    s, C = (ds.s_w.values, ds.Cs_w.values) if where == 'w' else (ds.s_rho.values, ds.Cs_r.values)
    hc = float(ds.hc)
    z0 = (hc * s[:, None, None] + h * C[:, None, None]) / (hc + h)
    return zeta + (zeta + h) * z0


class Run:
    def __init__(self, path):
        self.path = path
        self.ds = xr.open_dataset(path / 'roms_his.nc', decode_times=False)
        g = xr.open_dataset(path / 'grid.nc')
        self.j = self.ds.sizes['eta_rho'] // 2
        x, y = g.x_rho.values[self.j], g.y_rho.values[self.j]
        self.dist = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y))))) / 1000.0   # km
        self.dA = 1.0 / (g.pm.values * g.pn.values)
        self.hours = (self.ds.ocean_time.values - self.ds.ocean_time.values[0]) / 3600.0
        # cost: the CPU time ROMS reports ("Total:"), not the wall time, which includes any time the
        # machine (WSL) was suspended
        self.wall = np.nan
        for line in (path / 'roms.log').read_text(errors='replace').splitlines():
            if line.strip().startswith('Total:'):
                self.wall = float(line.split()[1])
                break
        meta = json.loads((path / 'run_meta.json').read_text())
        self.dt = float(meta['parameters']['DT'])

    def section(self, t, var='salt'):
        '''x edges (km), z of interfaces at x edges, values (N, xi) on the centreline.'''
        j = self.j
        zw = depths(self.ds, t)[:, j, :]
        xe = np.concatenate(([self.dist[0]], 0.5 * (self.dist[1:] + self.dist[:-1]), [self.dist[-1]]))
        zwe = np.concatenate((zw[:, :1], 0.5 * (zw[:, 1:] + zw[:, :-1]), zw[:, -1:]), axis=1)
        X = np.broadcast_to(xe, zwe.shape)
        return X, zwe, self.ds[var].isel(ocean_time=t).values[:, j, :]

    def index(self, hour):
        return int(np.argmin(np.abs(self.hours - hour)))

    def series(self):
        ds = self.ds
        inner = (slice(1, -1), slice(1, -1))
        A = self.dA[inner]
        rows = []
        for t in range(ds.sizes['ocean_time']):
            dz = np.diff(depths(ds, t), axis=0)[(slice(None),) + inner]
            dV = dz * A
            s = ds.salt.isel(ocean_time=t).values[(slice(None),) + inner]
            row = {'hour': self.hours[t], 'volume_m3': dV.sum(), 'salt_content': (s * dV).sum(),
                   'mean_dS': np.mean(s[0] - s[-1])}
            if 'dye_01' in ds:
                # the ventilation age is the last pair (dye_02 with NPT = 4, dye_03 with NPT = 6)
                last = max(int(v[4:6]) for v in ds.data_vars if v.startswith('dye_') and v[4:6].isdigit())
                c1 = ds.dye_01.isel(ocean_time=t).values[(slice(None),) + inner]
                c2 = ds[f'dye_{last:02d}'].isel(ocean_time=t).values[(slice(None),) + inner]
                a2 = ds[f'dye_{last:02d}_age'].isel(ocean_time=t).values[(slice(None),) + inner]
                row['river_water_m3'] = (c1 * dV).sum()
                row['ventilation_age_h'] = (a2 * dV).sum() / max((c2 * dV).sum(), 1e-12) / 3600.0
                row['bottom_ventilation_age_h'] = (a2[0] * dV[0]).sum() / max((c2[0] * dV[0]).sum(), 1e-12) / 3600.0
            rows.append(row)
        return pd.DataFrame(rows)


def draw_section(ax, run, t, title=None):
    X, Z, S = run.section(t)
    m = ax.pcolormesh(X, Z, np.clip(S, 0, 30), cmap='viridis', vmin=0, vmax=30, shading='flat')
    h = run.ds.h.values[run.j]
    ax.fill_between(run.dist, -h, -4.0, color=BOTTOM, lw=0)
    ax.set_ylim(-3.7, 0.6)
    ax.set_xlim(run.dist[0], run.dist[-1])
    if title:
        ax.set_title(title, fontsize=9, loc='left')
    return m


def frames_figure(runs, case):
    cols = [k for k in LAYERS if (k, case) in runs]
    fig, axes = plt.subplots(len(FRAME_HOURS), len(cols), figsize=(4.2 * len(cols), 1.55 * len(FRAME_HOURS)),
                             sharex=True, sharey=True, squeeze=False)
    for c, layer in enumerate(cols):
        run = runs[(layer, case)]
        for r, hour in enumerate(FRAME_HOURS):
            ax = axes[r, c]
            t = run.index(hour)
            m = draw_section(ax, run, t, f'{LAYERS[layer]}, {run.hours[t]:.0f} h' if r == 0 else f'{run.hours[t]:.0f} h')
            if c == 0:
                ax.set_ylabel('z (m)', fontsize=8)
            ax.tick_params(labelsize=7)
        axes[-1, c].set_xlabel('distance from the mouth (km)', fontsize=8)
    fig.suptitle(f'Alexander centreline salinity, {CASES[case]}', fontsize=10, x=0.01, ha='left')
    fig.colorbar(m, ax=axes, shrink=0.6, label='salinity')
    fig.savefig(OUT / f'frames_{case}.png', dpi=120)
    plt.close(fig)


def section_video(runs, case):
    cols = [k for k in LAYERS if (k, case) in runs]
    nt = min(runs[(k, case)].ds.sizes['ocean_time'] for k in cols)
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.2 * len(cols) + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]

    def update(t):
        for ax, layer in zip(axes, cols):
            ax.clear()
            run = runs[(layer, case)]
            draw_section(ax, run, t, f'{LAYERS[layer]}')
            ax.set_ylabel('z (m)', fontsize=8)
        axes[-1].set_xlabel('distance from the mouth (km)')
        fig.suptitle(f'Alexander centreline salinity, {CASES[case]}: {runs[(cols[0], case)].hours[t]:.0f} h',
                     fontsize=10, x=0.01, ha='left')
        return []

    update(0)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 30))
    fig.colorbar(sm, ax=list(axes), shrink=0.8, label='salinity')
    anim = animation.FuncAnimation(fig, update, frames=nt, blit=False)
    out = OUT / f'section_{case}.mp4'
    if animation.writers.is_available('ffmpeg'):
        anim.save(out, writer=animation.FFMpegWriter(fps=8, bitrate=1800), dpi=110)
    else:
        out = out.with_suffix('.gif')
        anim.save(out, writer=animation.PillowWriter(fps=8), dpi=80)
    plt.close(fig)
    return out


def timeseries_figure(series):
    metrics = [('mean_dS', 'bottom − surface salinity\n(interior mean)'),
               ('salt_content', 'salt content\n(relative to start)'),
               ('volume_m3', 'volume (m³)'),
               ('river_water_m3', 'river water in estuary (m³)'),
               ('ventilation_age_h', 'mean ventilation age (h)')]
    fig, axes = plt.subplots(len(metrics), len(CASES), figsize=(13, 2.2 * len(metrics)), sharex=True, squeeze=False)
    for c, case in enumerate(CASES):
        for (layer, cs), df in series.items():
            if cs != case:
                continue
            for r, (key, label) in enumerate(metrics):
                if key not in df:
                    continue
                y = df[key] / df[key].iloc[0] if key == 'salt_content' else df[key]
                axes[r, c].plot(df.hour, y, color=COLOURS[layer], label=LAYERS[layer], lw=1.5)
        axes[0, c].set_title(CASES[case], fontsize=10, loc='left')
        axes[-1, c].set_xlabel('hours')
    for r, (_, label) in enumerate(metrics):
        axes[r, 0].set_ylabel(label, fontsize=8)
    axes[0, 0].legend(fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'timeseries.png', dpi=120)
    plt.close(fig)


def main(runs_dir, video=True):
    global OUT
    OUT = Path(runs_dir) / 'output_figures'
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for p in sorted(Path(runs_dir).iterdir()):
        if (p / 'roms_his.nc').exists() and '_' in p.name:
            layer, case = p.name.split('_', 1)
            if layer in LAYERS and case in CASES:
                runs[(layer, case)] = Run(p)
    series = {k: r.series() for k, r in runs.items()}
    rows = []
    for (layer, case), df in series.items():
        run = runs[(layer, case)]
        end = df.iloc[-1]
        rows.append({'layers': layer, 'case': case, 'hours': end.hour, 'cpu_s': run.wall, 'DT_s': run.dt,
                     'cpu_min_per_sim_day': run.wall / 60 / (end.hour / 24) if end.hour else np.nan,
                     'mean_dS_end': end.mean_dS, 'salt_change_pct': 100 * (end.salt_content / df.salt_content.iloc[0] - 1),
                     'volume_change_pct': 100 * (end.volume_m3 / df.volume_m3.iloc[0] - 1),
                     'river_water_m3': end.get('river_water_m3', np.nan),
                     'ventilation_age_h': end.get('ventilation_age_h', np.nan),
                     'bottom_ventilation_age_h': end.get('bottom_ventilation_age_h', np.nan)})
    table = pd.DataFrame(rows).sort_values(['case', 'layers'])
    table.to_csv(OUT / 'summary.csv', index=False)
    print(table.round(3).to_string(index=False))
    timeseries_figure(series)
    for case in CASES:
        if any(c == case for _, c in runs):
            frames_figure(runs, case)
            if video:
                print('wrote', section_video(runs, case))
    print(f'figures in {OUT}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compare the vertical-layer runs')
    p.add_argument('--runs-dir', default=str(REPO.parent / 'ROMS_runs' / 'vertical_layers'))
    p.add_argument('--no-video', action='store_true')
    a = p.parse_args()
    main(a.runs_dir, not a.no_video)
