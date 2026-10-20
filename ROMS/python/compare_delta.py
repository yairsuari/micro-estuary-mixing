"""
Difference between the 20-layer and 10-layer runs of experiments/vertical_layers_20d.cfg.

Both runs use uniform layers (Vstretching 1, theta_s = theta_b = 0), so every
10-layer layer holds exactly two 20-layer layers. The 10-layer salinity is
interpolated linearly in the s coordinate (fraction of the local depth) onto the
20-layer levels, column by column, and |dS| = |S20 - S10| is computed there.

For each case (river1, river4) it writes, in ROMS_runs/vertical_layers_20d/output_figures/:
    delta_<case>.mp4    clip, one frame per 0.25 day: centreline section of S (20 layers)
                        and of |S20 - S10|
    delta_<case>.png    the same at 0.25, 1, 2, 5, 10, 15 and 20 days
and for both cases:
    delta_timeseries.png  volume-mean and 99th percentile of |dS| (whole interior), centreline
                          maximum, and the interior mean bottom-surface salinity of each run
    delta_summary.csv

Usage: python compare_delta.py [--runs-dir .../ROMS_runs/vertical_layers_20d] [--no-video]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

from common import REPO
from compare_layers import BOTTOM, depths

try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

CASES = {'river1': 'river 20 000 m³/day', 'river4': 'river 80 000 m³/day'}
FRAME_DAYS = (0.25, 1, 2, 5, 10, 15, 20)
DS_MAX = 3.0                      # colour scale of |dS|
OUT = REPO.parent / 'ROMS_runs' / 'vertical_layers_20d' / 'output_figures'     # set from --runs-dir in main


def interp_matrix(s_from, s_to):
    '''W (len(s_to), len(s_from)) for linear interpolation in s, constant beyond the end levels.'''
    W = np.zeros((s_to.size, s_from.size))
    for i, s in enumerate(s_to):
        if s <= s_from[0]:
            W[i, 0] = 1.0
        elif s >= s_from[-1]:
            W[i, -1] = 1.0
        else:
            k = np.searchsorted(s_from, s) - 1
            w = (s - s_from[k]) / (s_from[k + 1] - s_from[k])
            W[i, k], W[i, k + 1] = 1 - w, w
    return W


class Pair:
    def __init__(self, runs_dir, case):
        self.case = case
        self.d20 = xr.open_dataset(Path(runs_dir) / f'N20u_{case}' / 'roms_his.nc', decode_times=False)
        self.d10 = xr.open_dataset(Path(runs_dir) / f'N10u_{case}' / 'roms_his.nc', decode_times=False)
        for d in (self.d20, self.d10):
            assert int(d.Vstretching) == 1 and float(d.theta_s) == 0, 'uniform layers expected'
        self.W = interp_matrix(self.d10.s_rho.values, self.d20.s_rho.values)
        self.nt = min(self.d20.sizes['ocean_time'], self.d10.sizes['ocean_time'])
        self.days = (self.d20.ocean_time.values[:self.nt] - self.d20.ocean_time.values[0]) / 86400.0
        g = xr.open_dataset(Path(runs_dir) / f'N20u_{case}' / 'grid.nc')
        self.j = self.d20.sizes['eta_rho'] // 2
        x, y = g.x_rho.values[self.j], g.y_rho.values[self.j]
        self.dist = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y))))) / 1000.0
        self.dA = 1.0 / (g.pm.values * g.pn.values)
        self.h = self.d20.h.values

    def delta(self, t):
        '''S20 and |S20 - S10 interpolated to the 20 levels|, (20, eta, xi).'''
        s20 = self.d20.salt.isel(ocean_time=t).values
        s10 = self.d10.salt.isel(ocean_time=t).values
        s10_on20 = np.tensordot(self.W, s10, axes=(1, 0))
        return s20, np.abs(s20 - s10_on20)

    def section_mesh(self, t):
        zw = depths(self.d20, t)[:, self.j, :]
        xe = np.concatenate(([self.dist[0]], 0.5 * (self.dist[1:] + self.dist[:-1]), [self.dist[-1]]))
        zwe = np.concatenate((zw[:, :1], 0.5 * (zw[:, 1:] + zw[:, :-1]), zw[:, -1:]), axis=1)
        return np.broadcast_to(xe, zwe.shape), zwe

    def stats(self):
        inner = (slice(None), slice(1, -1), slice(1, -1))
        rows = []
        for t in range(self.nt):
            s20, ds = self.delta(t)
            dV = (np.diff(depths(self.d20, t), axis=0) * self.dA[None])[inner]
            s10 = self.d10.salt.isel(ocean_time=t).values
            rows.append({'day': self.days[t],
                         'mean_abs_dS': float((ds[inner] * dV).sum() / dV.sum()),
                         'p99_abs_dS': float(np.percentile(ds[inner], 99)),
                         'max_abs_dS_centreline': float(ds[:, self.j, 1:-1].max()),
                         'strat_N20': float(np.mean(s20[0, 1:-1, 1:-1] - s20[-1, 1:-1, 1:-1])),
                         'strat_N10': float(np.mean(s10[0, 1:-1, 1:-1] - s10[-1, 1:-1, 1:-1]))})
        return pd.DataFrame(rows)


def draw(ax_s, ax_d, pair, t, label=True):
    X, Z = pair.section_mesh(t)
    s20, ds = pair.delta(t)
    j = pair.j
    ms = ax_s.pcolormesh(X, Z, np.clip(s20[:, j, :], 0, 30), cmap='viridis', vmin=0, vmax=30, shading='flat')
    md = ax_d.pcolormesh(X, Z, ds[:, j, :], cmap='magma_r', vmin=0, vmax=DS_MAX, shading='flat')
    for ax in (ax_s, ax_d):
        ax.fill_between(pair.dist, -pair.h[j], -4.0, color=BOTTOM, lw=0)
        ax.set_ylim(-3.7, 0.6)
        ax.set_xlim(pair.dist[0], pair.dist[-1])
        ax.tick_params(labelsize=7)
    if label:
        ax_s.set_title('salinity, 20 layers', fontsize=9, loc='left')
        ax_d.set_title('|S(20 layers) − S(10 layers)|', fontsize=9, loc='left')
    return ms, md


def clip(pair):
    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ms, md = draw(ax_s, ax_d, pair, 0)
    fig.colorbar(ms, ax=ax_s, label='salinity')
    fig.colorbar(md, ax=ax_d, label='|ΔS|', extend='max')

    def update(t):
        for ax in (ax_s, ax_d):
            for coll in list(ax.collections):
                coll.remove()
        draw(ax_s, ax_d, pair, t)
        ax_d.set_xlabel('distance from the mouth (km)')
        ax_s.set_ylabel('z (m)')
        ax_d.set_ylabel('z (m)')
        fig.suptitle(f'Alexander centreline, {CASES[pair.case]}: day {pair.days[t]:.2f}', x=0.01, ha='left', fontsize=10)
        return []

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=pair.nt, blit=False)
    out = OUT / f'delta_{pair.case}.mp4'
    if animation.writers.is_available('ffmpeg'):
        anim.save(out, writer=animation.FFMpegWriter(fps=6, bitrate=1800), dpi=110)
    else:
        out = out.with_suffix('.gif')
        anim.save(out, writer=animation.PillowWriter(fps=6), dpi=80)
    plt.close(fig)
    return out


def frames(pair):
    days = [d for d in FRAME_DAYS if d <= pair.days[-1] + 1e-6]
    fig, axes = plt.subplots(len(days), 2, figsize=(12, 1.6 * len(days) + 0.8), sharex=True, sharey=True, squeeze=False)
    for r, d in enumerate(days):
        t = int(np.argmin(np.abs(pair.days - d)))
        ms, md = draw(axes[r, 0], axes[r, 1], pair, t, label=(r == 0))
        axes[r, 0].set_ylabel(f'day {pair.days[t]:g}\nz (m)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('distance from the mouth (km)', fontsize=8)
    fig.colorbar(ms, ax=axes[:, 0], shrink=0.5, label='salinity')
    fig.colorbar(md, ax=axes[:, 1], shrink=0.5, label='|ΔS|', extend='max')
    fig.suptitle(f'Alexander centreline, {CASES[pair.case]}', x=0.01, ha='left', fontsize=10)
    fig.savefig(OUT / f'delta_{pair.case}.png', dpi=110)
    plt.close(fig)


def main(runs_dir, video=True):
    global OUT
    OUT = Path(runs_dir) / 'output_figures'
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for case in CASES:
        if not (Path(runs_dir) / f'N20u_{case}' / 'roms_his.nc').exists():
            continue
        pair = Pair(runs_dir, case)
        tables[case] = pair.stats()
        frames(pair)
        if video:
            print('wrote', clip(pair))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for case, df in tables.items():
        ls = '-' if case == 'river1' else '--'
        axes[0].plot(df.day, df.mean_abs_dS, ls, color='#1f5fa8', label=f'{CASES[case]}: volume mean')
        axes[0].plot(df.day, df.p99_abs_dS, ls, color='#c0392b', label=f'{CASES[case]}: 99th percentile')
        axes[1].plot(df.day, df.max_abs_dS_centreline, ls, color='k', label=CASES[case])
        axes[2].plot(df.day, df.strat_N20, ls, color='#2a9d4b', label=f'{CASES[case]}, 20 layers')
        axes[2].plot(df.day, df.strat_N10, ls, color='#e08a1e', label=f'{CASES[case]}, 10 layers')
    axes[0].set(title='|S20 − S10|, whole interior', xlabel='day', ylabel='|ΔS|')
    axes[1].set(title='|S20 − S10|, centreline maximum', xlabel='day', ylabel='|ΔS|')
    axes[2].set(title='bottom − surface salinity (interior mean)', xlabel='day')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / 'delta_timeseries.png', dpi=120)
    plt.close(fig)
    table = pd.concat({c: df for c, df in tables.items()}, names=['case']).reset_index(level=0)
    table.to_csv(OUT / 'delta_summary.csv', index=False)
    for case, df in tables.items():
        print(case)
        print(df.iloc[[1, 4, 8, 20, 40, 60, -1]].round(3).to_string(index=False))
    print(f'figures in {OUT}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='|S20 - S10| for the 20-day layer runs')
    p.add_argument('--runs-dir', default=str(REPO.parent / 'ROMS_runs' / 'vertical_layers_20d'))
    p.add_argument('--no-video', action='store_true')
    a = p.parse_args()
    main(a.runs_dir, not a.no_video)
