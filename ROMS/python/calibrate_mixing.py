"""
Calibration of the background salt diffusivity AKS_BAK against the observed mean daily exchange
rate of the salt pool at A4 (E1).

Target: the mean of the daily exchange rates of all fresh-water-only periods (mixing_periods.py,
input/daily_exchange.csv): 3.6 %/day, 95% interval 3.1-4.1.

Only AKS_BAK is calibrated (AKT_BAK = AKS_BAK, no effect with uniform temperature). The background
viscosity is fixed at AKV_BAK = 1e-6 m2/s (molecular). Fixed forcing: Alexander, default ini.nc,
river 17 000 m3/day (RIVER_SCALE = 0.85 of rivers.nc), uniform 2 m/s west wind (wind_w2.nc, wind
build), sea level 0; 16 days, hourly history.

Model score: the A4 sensors (0.2 m below the surface and above the bed, analyse_runs.py) give, for
each day 2-16, r_d = -(slope of the hourly bottom salinity of that day) / (mean bottom - surface of
that day), as for the observations; the score is the mean of r_d.

Commands
    round K1 K2 ...      write ROMS/scripts/experiments/mixing_calibration_r<n>.cfg for these
                         AKS_BAK values (run it with run_experiment.sh in WSL)
    score                score every finished run of ROMS_runs/mixing_calibration, write
                         calibration_log.csv, daily_exchange_model.csv and
                         output_figures/calibration.png, and print the next bracket
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import REPO
from analyse_runs import RUNS_DIR, run_series, cpu_seconds

EXPERIMENT = 'mixing_calibration'
EXP = RUNS_DIR / EXPERIMENT
EXPERIMENTS_DIR = REPO / 'ROMS' / 'scripts' / 'experiments'
FIGURES = EXP / 'output_figures'
OBSERVED = EXP / 'input' / 'daily_exchange.csv'
FIRST_DAY, LAST_DAY = 1, 16                    # days 2-16 (day 1 = adjustment to river and wind)
COMMON = ('DAYS=16 HIS_HOURS=1 AVG_HOURS=24 DIA_HOURS=24 RIVERS=on RIVER_SCALE=0.85 '
          'WIND_FILE=wind_w2.nc EXEC=micro_estuary_my25_serial_dyes_wind AKV_BAK=1e-6')


def target():
    v = pd.read_csv(OBSERVED).exchange.dropna()
    ci = 1.96 * v.std() / np.sqrt(len(v))
    return v.mean(), v.mean() - ci, v.mean() + ci


def run_name(k):
    return f'K{k:.3e}'.replace('e-0', 'e-')


def write_round(values):
    n = 1 + max([int(p.stem.rsplit('_r', 1)[1]) for p in EXPERIMENTS_DIR.glob(f'{EXPERIMENT}_r*.cfg')] or [0])
    lines = [f'# E1 vertical-mixing calibration, round {n} (written by ROMS/python/calibrate_mixing.py).',
             '# Only AKS_BAK varies (AKT_BAK = AKS_BAK); AKV_BAK = 1e-6 fixed; river 17 000 m3/day, west wind',
             '# 2 m/s, sea level 0, default ini.nc, 16 days. Target: mean daily A4 exchange rate 3.6 %/day.',
             f'EXPERIMENT={EXPERIMENT}', f'PARALLEL={min(4, len(values))}', f'COMMON {COMMON}', '']
    for k in values:
        lines.append(f'{run_name(k):14s} Alexander  AKS_BAK={k:.4g} AKT_BAK={k:.4g}')
    path = EXPERIMENTS_DIR / f'{EXPERIMENT}_r{n}.cfg'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')      # LF: read by bash
    print(path)


def daily_exchange(series, station='A4'):
    '''daily exchange rates (%/day) from the hourly sensor series of a run.'''
    rows = []
    for d in range(FIRST_DAY, LAST_DAY):
        w = series[(series.day >= d) & (series.day <= d + 1)]
        if len(w) < 3:
            continue
        slope = np.polyfit(w.day, w[f'{station}_bottom'], 1)[0]
        ds = float((w[f'{station}_bottom'] - w[f'{station}_surface']).mean())
        rows.append({'day': d + 1, 'slope': slope, 'dS': ds, 'exchange': -100 * slope / ds if ds > 0 else np.nan})
    return pd.DataFrame(rows)


def score():
    tgt, lo, hi = target()
    log, daily, series_all = [], [], {}
    for path in sorted(EXP.glob('K*')):
        if not (path / 'roms_his.nc').exists() or 'ROMS: DONE' not in (path / 'roms.log').read_text(errors='replace'):
            continue
        meta = json.loads((path / 'run_meta.json').read_text())['parameters']
        s = run_series(path)
        series_all[path.name] = s
        row = {'run': path.name, 'AKS_BAK': float(meta['AKS_BAK']), 'AKV_BAK': float(meta['AKV_BAK'])}
        for st in ('A4', 'A3'):
            d = daily_exchange(s, st)
            d.insert(0, 'station', st)
            d.insert(0, 'run', path.name)
            daily.append(d)
            row[f'{st}_exchange_mean'] = d.exchange.mean()
            row[f'{st}_exchange_median'] = d.exchange.median()
            row[f'{st}_dS_day2'] = d.dS.iloc[0] if len(d) else np.nan
            row[f'{st}_dS_day16'] = d.dS.iloc[-1] if len(d) else np.nan
        row['K_eff_days2_16'] = float(np.nanmean(s.K_eff[(s.day > FIRST_DAY) & (s.day <= LAST_DAY)]))
        row['nan'] = int(s.nan.sum())
        row['cpu_hours'] = cpu_seconds(path) / 3600
        log.append(row)
    if not log:
        sys.exit('no finished runs')
    log = pd.DataFrame(log).sort_values('AKS_BAK')
    log['in_target'] = log.A4_exchange_mean.between(lo, hi)
    log.to_csv(EXP / 'calibration_log.csv', index=False, float_format='%.4g')
    pd.concat(daily).to_csv(EXP / 'daily_exchange_model.csv', index=False, float_format='%.4g')
    with pd.option_context('display.width', 200):
        print(log.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(f'target {tgt:.2f} %/day ({lo:.2f}-{hi:.2f})')
    estimate = interpolate(log, tgt, lo, hi)
    figure(log, pd.concat(daily), series_all, tgt, lo, hi, estimate)
    return log


def interpolate(log, tgt, lo, hi):
    '''AKS_BAK at the target from log(exchange) against log(K), and the next bracket.'''
    log = log[log.AKS_BAK > 0]                                   # K = 0 (no background) is a diagnostic only
    x, y = np.log10(log.AKS_BAK.values), log.A4_exchange_mean.values
    ok = y > 0
    x, y = x[ok], np.log10(y[ok])
    if len(x) < 2:
        return None
    above = np.where(10 ** y >= tgt)[0]
    below = np.where(10 ** y < tgt)[0]
    if len(above) == 0 or len(below) == 0:
        side = 'above' if len(below) == 0 else 'below'
        print(f'target not bracketed: every run exchanges {side} the target; extend the range')
        return None
    # bracket: the closest runs on either side
    ia = above[np.argmin(np.abs(y[above] - np.log10(tgt)))]
    ib = below[np.argmin(np.abs(y[below] - np.log10(tgt)))]
    (x0, y0), (x1, y1) = sorted([(x[ia], y[ia]), (x[ib], y[ib])])
    est = {}
    for name, t in (('K', tgt), ('K_lo', lo), ('K_hi', hi)):
        est[name] = 10 ** (x0 + (np.log10(t) - y0) * (x1 - x0) / (y1 - y0))
    nxt = 10 ** (x0 + (x1 - x0) * np.arange(1, 5) / 5)
    print(f'bracket {10 ** x0:.3g}-{10 ** x1:.3g}; interpolated AKS_BAK at the target {est["K"]:.3g} '
          f'(interval {min(est["K_lo"], est["K_hi"]):.3g}-{max(est["K_lo"], est["K_hi"]):.3g})')
    print('next round: python calibrate_mixing.py round ' + ' '.join(f'{k:.3g}' for k in nxt))
    return est


def figure(log, daily, series_all, tgt, lo, hi, est):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    a = axes[0]
    a.axhspan(lo, hi, color='#f2c46b', alpha=0.5, lw=0, label=f'observed mean, 95% ({lo:.1f}–{hi:.1f})')
    a.axhline(tgt, color='#8a5a00', lw=1)
    pos = log[log.AKS_BAK > 0]
    a.plot(pos.AKS_BAK, pos.A4_exchange_mean, 'o-', color='#1f5fa8', label='model, A4 (mean of days 2–16)')
    a.plot(pos.AKS_BAK, pos.A3_exchange_mean, 's--', color='#2a9d4b', ms=4, label='model, A3')
    for _, r in log[log.AKS_BAK == 0].iterrows():
        a.axhline(r.A4_exchange_mean, color='#1f5fa8', ls=':', lw=1, label='model, A4, AKS_BAK = 0')
    if est:
        a.axvline(est['K'], color='#b2392f', lw=1.2, label=f'AKS_BAK at target ≈ {est["K"]:.2g}')
    a.set_xscale('log')
    a.set_xlabel('AKS_BAK (m²/s)')
    a.set_ylabel('daily exchange rate (%/day)')
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which='both')
    a = axes[1]
    obs = pd.read_csv(OBSERVED).exchange.dropna()
    bins = np.arange(-6, 26, 2)
    a.hist(obs.clip(-6, 25), bins=bins, density=True, color='0.75', label=f'observed ({len(obs)} days)')
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(log)))
    for c, run in zip(cols, log.run):
        v = daily[(daily.run == run) & (daily.station == 'A4')].exchange
        a.hist(v.clip(-6, 25), bins=bins, density=True, histtype='step', color=c, lw=1.5, label=run)
    a.set_xlabel('daily exchange rate at A4 (%/day)')
    a.set_ylabel('density')
    a.legend(fontsize=6)
    a = axes[2]
    for c, run in zip(cols, log.run):
        s = series_all[run]
        a.plot(s.day, s.A4_bottom, color=c, lw=1.2, label=run)
        a.plot(s.day, s.A4_surface, color=c, lw=0.8, ls=':')
    a.axvspan(0, FIRST_DAY, color='0.85', lw=0)
    a.set_xlabel('day')
    a.set_ylabel('A4 salinity (bottom solid, surface dotted)')
    a.legend(fontsize=6)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / 'calibration.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='E1 calibration of AKS_BAK')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('round')
    r.add_argument('values', type=float, nargs='+')
    sub.add_parser('score')
    a = ap.parse_args()
    if a.cmd == 'round':
        write_round(a.values)
    else:
        score()
