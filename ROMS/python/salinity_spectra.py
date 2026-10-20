"""
Spectra of the observed salinity at A4 and A3 (surface and bottom sensors), to see how strong the diurnal
cycle is (sea breeze) before the wind / internal-wave experiment.

Method
    1. Despiking as in mixing_periods.py (running median of 5 samples), 10-min means, gaps of up to
       1 h filled linearly.
    2. Welch periodogram on every continuous stretch of record: segments of SEG_DAYS (8) days, 50%
       overlap, Hann window, linear detrend; the segment spectra are averaged over the whole record,
       separately for the dry (May-Oct) and wet (Nov-Apr) season.
    3. Diurnal band: the variance between 0.9 and 1.1 cycles/day, as a fraction of the variance between
       0.5 and 6 cycles/day, and the peak-to-background ratio at 1 cycle/day (background = median
       spectral level in 0.6-0.85 and 1.15-1.4 cycles/day).
    4. Mean daily cycle: deviation from the daily mean, averaged by hour of the day (local time, as
       exported), and its half range.

Writes ROMS_runs/wind_iw/output_figures/manuscript_plot3.png (Fig. 3 of the manuscript), salinity_spectra.csv (spectra) and
salinity_diurnal.csv (diurnal band statistics) and salinity_daily_cycle.csv (mean daily cycles, local hour, with the half width of their 90% confidence
interval across days).

Usage: python salinity_spectra.py
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mixing_periods import load_station, despike, FILES
from analyse_runs import RUNS_DIR

OUT = RUNS_DIR / 'wind_iw' / 'output_figures'
STEP = '10min'
PER_DAY = 144
SEG_DAYS = 8
COLOURS = {'surface': '#1f5fa8', 'bottom': '#b2392f'}


def regular(name):
    d = despike(load_station(name), 5).resample(STEP).mean()
    return d.interpolate(limit=6, limit_area='inside')


def welch(series):
    '''average Welch spectrum over the continuous stretches of a series with gaps.'''
    nper = SEG_DAYS * PER_DAY
    ok = series.notna().to_numpy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(int), [0]))))
    total, weight, f = 0.0, 0, None
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < nper:
            continue
        f, p = signal.welch(series.to_numpy()[a:b], fs=PER_DAY, nperseg=nper, noverlap=nper // 2,
                            window='hann', detrend='linear')
        n = 1 + (b - a - nper) // (nper // 2)
        total, weight = total + p * n, weight + n
    return (f, total / weight, weight) if weight else (None, None, 0)


def band_stats(f, p):
    df = f[1] - f[0]
    var = lambda lo, hi: p[(f >= lo) & (f <= hi)].sum() * df
    bg = np.median(p[((f >= 0.6) & (f <= 0.85)) | ((f >= 1.15) & (f <= 1.4))])
    peak = p[(f >= 0.95) & (f <= 1.05)].max()
    return {'diurnal_fraction': var(0.9, 1.1) / var(0.5, 6.0), 'peak_to_background': peak / bg,
            'semidiurnal_fraction': var(1.9, 2.1) / var(0.5, 6.0)}


def daily_cycle(series):
    s = series.dropna()
    anom = s - s.groupby(s.index.floor('D')).transform('mean')
    return anom.groupby(anom.index.hour).mean()


def daily_cycle_ci(series, level=0.90):
    '''half width of the confidence interval of the mean daily cycle, by hour, from the hourly means of the
    individual days (days treated as independent)'''
    s = series.dropna()
    anom = s - s.groupby(s.index.floor('D')).transform('mean')
    per_day = anom.groupby([anom.index.floor('D'), anom.index.hour]).mean()
    g = per_day.groupby(level=1)
    n = g.count()
    return stats.t.ppf(0.5 + level / 2, n - 1) * g.std() / np.sqrt(n), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spectra, stats, cycles = [], [], {}
    for name in FILES:
        d = regular(name)
        dry = d.index.month.isin(range(5, 11))
        for sensor in ('surface', 'bottom'):
            for season, mask in (('dry', dry), ('wet', ~dry)):
                f, p, n = welch(d[sensor].where(mask))
                if f is None:
                    continue
                spectra.append(pd.DataFrame({'station': name, 'sensor': sensor, 'season': season,
                                             'freq_cpd': f, 'psd': p}))
                cyc = daily_cycle(d[sensor].where(mask))
                cycles[(name, sensor, season)] = (cyc, *daily_cycle_ci(d[sensor].where(mask)))
                stats.append({'station': name, 'sensor': sensor, 'season': season, 'segments': n,
                              **band_stats(f, p), 'daily_half_range': (cyc.max() - cyc.min()) / 2,
                              'hour_of_max': int(cyc.idxmax()), 'hour_of_min': int(cyc.idxmin())})
    spectra = pd.concat(spectra)
    stats = pd.DataFrame(stats)
    spectra.to_csv(OUT / 'salinity_spectra.csv', index=False, float_format='%.5g')
    stats.to_csv(OUT / 'salinity_diurnal.csv', index=False, float_format='%.4g')
    pd.DataFrame([{'station': st, 'sensor': se, 'season': sea, 'hour': int(h), 'anomaly': v, 'ci90': ci[h],
                   'n_days': int(n[h])}
                  for (st, se, sea), (cyc, ci, n) in cycles.items() for h, v in cyc.items()]).to_csv(
        OUT / 'salinity_daily_cycle.csv', index=False, float_format='%.5g')
    print(stats.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

    # (a) and (b) share the y axis (ticks and label only on (a)), so the font can be larger
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(16, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.22)
    left = gs[0].subgridspec(1, 2, wspace=0.0)
    a0 = fig.add_subplot(left[0])
    axes = [a0, fig.add_subplot(left[1], sharey=a0), fig.add_subplot(gs[1])]
    for a, name, letter in zip(axes[:2], FILES, 'ab'):
        for sensor in ('surface', 'bottom'):
            for season, ls in (('dry', '-'), ('wet', ':')):
                s = spectra[(spectra.station == name) & (spectra.sensor == sensor) & (spectra.season == season)]
                if len(s):
                    a.loglog(s.freq_cpd, s.psd, color=COLOURS[sensor], ls=ls, lw=1.2,
                             label=f'{sensor}, {season} season')
        for k in (1, 2, 3):
            a.axvline(k, color='0.6', lw=0.8, ls='--')
        a.set_xlim(1 / SEG_DAYS, PER_DAY / 2)
        a.set_xlabel('frequency (cycles/day)')
        a.text(0.04, 0.04, f'({letter}) {name}', transform=a.transAxes, fontsize=16, fontweight='bold')
        a.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('PSD (PSU² per cycle/day)')
    axes[1].tick_params(labelleft=False)
    axes[1].set_xticks([1, 10])                      # avoid the touching 10^-1 / 10^1 labels at the shared edge
    axes[0].legend(fontsize=12, loc='upper right')
    a = axes[2]
    for (name, sensor, season), (cyc, _ci, _n) in cycles.items():
        if season != 'dry':
            continue
        a.plot(cyc.index, cyc.values, color=COLOURS[sensor], ls='-' if name == 'A4' else '--', marker='o', ms=3,
               label=f'{name} {sensor}')
    a.axhline(0, color='0.5', lw=0.8)
    a.set_xticks(range(0, 24, 3))
    a.set_xlabel('hour of day (local time)')
    a.set_ylabel('salinity − daily mean (PSU)')
    a.text(0.97, 0.04, '(c) dry season', transform=a.transAxes, fontsize=16, fontweight='bold', ha='right')
    a.legend(fontsize=12)
    a.grid(alpha=0.3)
    fig.savefig(OUT / 'manuscript_plot3.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
