"""
E4a: wind variability and internal waves, on the three domains. Compares
the paired runs of ROMS_runs/<experiment> (wind_iw_bar, the closed mouth; wind_iw, the old mouth): diurnal sine wind (diurnal_M) and constant wind (const_M) with the same
mean |stress|, M = 1, 2, 4, 6, 8 m/s, and the windless reference (none), on Alexander (runs without prefix,
wind from 270 deg), Meander (meander_*) and NoMeander (nomeander_*), both with the wind from 231 deg.
River 20 000 m3/day, 30 days, hourly history.

Seawater over the bar. The diurnal wind drives seawater (S = 39.5, tracer dye_02) into the estuary over the
mouth. Salt is linear in the water fractions, so the salt of the original pool alone is
    S_orig = S - S_SEA * dye_02,
which obeys the same equations as S with zero salt from the sea. Exchange rates of S_orig measure the loss of
the original pool by mixing; exchange rates of S include the resupply (as a sensor in the field would).

Per run (days 2-30):
    exchange rates   mean daily exchange rate at A4 and A3 (calibrate_mixing.daily_exchange), of S_orig and of
                     S; days with a bottom - surface difference below DS_MIN (as mixing_periods.py) are skipped,
                     since the rate blows up as S_b - S_s -> 0
    pool rate        whole-estuary salt loss rate, 100 (1 - exp(-k)) %/day, k the slope of -ln(salt content)
                     against time from day FIRST until POOL_END of it is left (at least 2 days), i.e. the % of the
                     salt lost per day of an exponential fit; unlike the daily station rates it also works when the
                     pool goes in days. pool_rate_orig: the same for the original pool (S_orig)
    K_eff            effective vertical diffusivity of the whole estuary (analyse_runs.run_series); in the CSV
                     only, since it is not defined once the pool is nearly gone
    salt left        salt content at day 30 / day 0, of S and of S_orig
    seawater         seawater volume in the estuary at day 30 (m3), and its net rate of entry (m3/day)
    halocline        depth of the halocline at A4 and A3: the depth of the largest stable dS/dz (the definition of
                     Suari et al.), refined by a parabola through the three levels around the maximum, since the
                     10 layers would otherwise give steps of 0.3-0.4 m; mean, and half range of its mean daily cycle
    sensor cycles    half range of the mean daily cycle of the sensor salinity (surface, bottom)
    spectra          diurnal fraction and peak-to-background of the sensor salinity, as salinity_spectra.py

The hourly series of each run are cached in output_figures/cache/ (rebuilt when roms_his.nc is newer).

Writes ROMS_runs/<experiment>/output_figures/: wind_iw.csv, wind_iw_mixing.png, wind_iw_internal.png,
wind_iw_series.png, wind_iw_seawater_cycle.png, wind_iw_spectra.png; and the manuscript figure
docs/figures/manuscript_plot4.png (Fig. 4).

Usage: python compare_wind_iw.py [--jobs 6] [--experiment wind_iw_bar]
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import calibrate_mixing as cm
from analyse_runs import REPO, RUNS_DIR, run_series, sensor_values, station_cells
from common import estuary_slice
from compare_layers import depths
from salinity_spectra import band_stats

OBS = RUNS_DIR / 'wind_iw' / 'output_figures'     # observed spectra and daily cycles (salinity_spectra.py)


def set_experiment(name):
    '''point EXP, OUT and CACHE at ROMS_runs/<name> (also in the Pool workers, which do not share globals).'''
    global EXP, OUT, CACHE
    EXP = RUNS_DIR / name
    OUT = EXP / 'output_figures'
    CACHE = OUT / 'cache'


set_experiment('wind_iw_bar')
SPEEDS = (1, 2, 4, 6, 8)
DOMAINS = {'Alexander': '', 'Meander': 'meander_', 'NoMeander': 'nomeander_'}
FIRST, LAST = 1, 30
S_SEA = 39.5               # salinity of the sea boundary (bry.nc salt_west)
DS_MIN = 3.0               # PSU, smallest bottom - surface difference for a daily exchange rate
G_MIN = 1.0                # PSU/m, smallest max dS/dz counted as a halocline
STRAT_MIN = 1.0            # PSU, the internal-wave measures use the days until the daily mean S_b - S_s first
                           # falls below this (afterwards the pool is gone and the salinity cycles mean nothing)
LAG_MIN = 0.05             # PSU, the lag of the daily maxima is shown only when both daily half ranges reach this
SPEC_DAYS = 8              # days of stratified record needed for a spectrum (one Welch segment)
POOL_END = 0.05            # the pool rate is fitted until this fraction of the original pool is left
OBS_C, OBS_CI_C = '#945099', '#5e2f63'     # observed: purple, unlike any domain colour
DC = {'Alexander': '#1f5fa8', 'Meander': '#e08a1e', 'NoMeander': '#2a9d4b'}
LS = {'const': '-', 'diurnal': '--'}
SENSOR_C = {'surface': '#1f5fa8', 'bottom': '#b2392f'}


def finished(path):
    log = path / 'roms.log'
    return (path / 'roms_his.nc').exists() and log.exists() and 'ROMS: DONE' in log.read_text(errors='replace')


def extras(path):
    '''hourly halocline depth, seawater volume, original-pool salt and S_orig at the sensors.'''
    ds = xr.open_dataset(path / 'roms_his.nc', decode_times=False)
    grid = xr.open_dataset(path / 'grid.nc')
    dA = 1.0 / (grid.pm.values * grid.pn.values)
    j, cells = station_cells(grid)
    inner = estuary_slice(grid, 3)
    rows = []
    for t in range(ds.sizes['ocean_time']):
        zr, zw = depths(ds, t, 'r'), depths(ds, t)
        dV = (np.diff(zw, axis=0) * dA)[inner]
        s = ds.salt.isel(ocean_time=t).values
        c2 = ds.dye_02.isel(ocean_time=t).values
        so = s - S_SEA * c2
        row = {'seawater_m3': float((c2[inner] * dV).sum()), 'orig_salt_content': float((so[inner] * dV).sum())}
        for name, i in cells.items():
            z = zr[:, j, i]
            g = np.maximum(-np.diff(s[:, j, i]) / np.diff(z), 0.0)     # -dS/dz, stable part
            zm = 0.5 * (z[1:] + z[:-1])
            row[f'{name}_halocline'] = (float(zw[-1, j, i] - max_gradient_z(zm, g)) if g.max() >= G_MIN
                                        else np.nan)                 # no halocline in a mixed column
            row[f'{name}_orig_surface'], row[f'{name}_orig_bottom'] = sensor_values(so, zr, zw, j, i)
        rows.append(row)
    return pd.DataFrame(rows)


def max_gradient_z(zm, g):
    '''level of the largest gradient g, refined by a parabola through it and its two neighbours'''
    k = int(np.argmax(g))
    if 0 < k < len(g) - 1:
        a, b, c = g[k - 1], g[k], g[k + 1]
        den = a - 2 * b + c
        if den < 0:
            off = 0.5 * (a - c) / den                       # in units of the level spacing, within +-0.5
            dz = (zm[k + 1] - zm[k - 1]) / 2
            return zm[k] + off * dz
    return zm[k]


def series(path):
    '''hourly series of a run (run_series + extras), cached.'''
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{path.name}.csv'
    if f.exists() and f.stat().st_mtime > (path / 'roms_his.nc').stat().st_mtime:
        return pd.read_csv(f)
    s = pd.concat([run_series(path), extras(path)], axis=1)
    s.to_csv(f, index=False, float_format='%.7g')
    return s


def daily_half_range(day, values):
    s = pd.Series(np.asarray(values), index=np.asarray(day))
    s = s[(s.index >= FIRST) & (s.index < LAST)]
    d = np.floor(s.index.to_numpy())
    anom = s - s.groupby(d).transform('mean')
    hour = np.round((s.index.to_numpy() - d) * 24).astype(int) % 24
    g = anom.groupby(hour)
    cyc = g.mean().where(g.count() >= 3)
    if cyc.notna().sum() < 20:
        return np.nan, cyc
    return (cyc.max() - cyc.min()) / 2, cyc


def stratified(s, st):
    '''the part of the record (from FIRST) before the daily mean S_b - S_s first falls below STRAT_MIN'''
    w = s[(s.day >= FIRST) & (s.day < LAST)]
    ds = (w[f'{st}_bottom'] - w[f'{st}_surface']).groupby(np.floor(w.day)).mean()
    weak = ds.index[ds < STRAT_MIN]
    return w[w.day < weak.min()] if len(weak) else w


def lag_hours(c1, c2):
    '''hours between the maxima of two mean daily cycles, 0-12 (12 = antiphase)'''
    if c1.notna().sum() < 20 or c2.notna().sum() < 20:
        return np.nan
    d = abs(int(c1.idxmax()) - int(c2.idxmax())) % 24
    return float(min(d, 24 - d))


def spectrum(day, values):
    s = np.asarray(values)[(np.asarray(day) >= FIRST)]
    f, p = signal.welch(s, fs=24, nperseg=8 * 24, noverlap=4 * 24, window='hann', detrend='linear')
    return f, p


def exchange(s, st, orig):
    '''mean daily exchange rate (%/day) and the number of days with S_b - S_s >= DS_MIN.'''
    if orig:
        s = s.assign(**{f'{st}_bottom': s[f'{st}_orig_bottom'], f'{st}_surface': s[f'{st}_orig_surface']})
    cm.FIRST_DAY, cm.LAST_DAY = FIRST, LAST
    d = cm.daily_exchange(s, st)
    d = d[d.dS >= DS_MIN]
    return d.exchange.mean(), len(d)


def analyse(path):
    s = series(path)
    row = {}
    for st in ('A4', 'A3'):
        row[f'{st}_exchange_orig'], row[f'{st}_days_orig'] = exchange(s, st, True)
        row[f'{st}_exchange'], row[f'{st}_days'] = exchange(s, st, False)
        w = stratified(s, st)
        row[f'{st}_stratified_days'] = float(w.day.iloc[-1] - w.day.iloc[0]) if len(w) else 0.0
        row[f'{st}_halocline_mean'] = w[f'{st}_halocline'].mean()
        row[f'{st}_halocline_halfrange'] = daily_half_range(w.day, w[f'{st}_halocline'])[0]
        cycles = {}
        for sensor in ('surface', 'bottom'):
            row[f'{st}_{sensor}_halfrange'], cycles[sensor] = daily_half_range(w.day, w[f'{st}_{sensor}'])
            if row[f'{st}_stratified_days'] >= SPEC_DAYS:
                f, p = spectrum(w.day, w[f'{st}_{sensor}'])
                b = band_stats(f, p)
            else:
                b = {'diurnal_fraction': np.nan, 'peak_to_background': np.nan}
            row[f'{st}_{sensor}_diurnal_fraction'] = b['diurnal_fraction']
            row[f'{st}_{sensor}_peak_to_background'] = b['peak_to_background']
        weak = min(row[f'{st}_surface_halfrange'], row[f'{st}_bottom_halfrange']) < LAG_MIN
        row[f'{st}_surface_bottom_lag_h'] = np.nan if weak else lag_hours(cycles['surface'], cycles['bottom'])
    for col, key in (('salt_content', 'pool_rate'), ('orig_salt_content', 'pool_rate_orig')):
        w = s[s.day >= FIRST]
        c = w[col] / s[col].iloc[0]
        end = w.day[c < POOL_END].min() if (c < POOL_END).any() else LAST
        w = w[w.day <= max(end, FIRST + 2)]
        k = -np.polyfit(w.day, np.log(np.maximum(w[col], 1e-12)), 1)[0]
        row[key] = 100 * (1 - np.exp(-k))
    row['K_eff'] = float(np.nanmean(s.K_eff[s.day > FIRST]))
    row['salt_left'] = s.salt_content.iloc[-1] / s.salt_content.iloc[0]
    row['orig_salt_left'] = s.orig_salt_content.iloc[-1] / s.orig_salt_content.iloc[0]
    row['seawater_m3'] = s.seawater_m3.iloc[-1]
    w = s[s.day >= FIRST]
    row['seawater_m3_per_day'] = np.polyfit(w.day, w.seawater_m3, 1)[0]
    row['nan'] = int(s.nan.sum())
    return row


def runs():
    out = []
    for domain, prefix in DOMAINS.items():
        out.append((domain, 'none', 0, f'{prefix}none'))
        out += [(domain, k, m, f'{prefix}{k}_{m}') for m in SPEEDS for k in ('const', 'diurnal')]
    return out


def work(item):
    domain, wind, m, name = item
    path = EXP / name
    if not finished(path):
        return None
    return {'run': name, 'domain': domain, 'wind': wind, 'M': m, **analyse(path)}


def pairs(ax, t, col, ylabel, observed=None, log=False, legend=True):
    for domain in DOMAINS:
        for k in ('const', 'diurnal'):
            d = t[(t.domain == domain) & (t.wind == k)]
            ax.plot(d.M, d[col], marker='o', ms=4, ls=LS[k], color=DC[domain], label=f'{domain}, {k}')
        n = t[(t.domain == domain) & (t.wind == 'none')]
        if len(n):
            ax.axhline(n[col].iloc[0], color=DC[domain], ls=':', lw=1)
    if observed is not None:
        ax.axhspan(*observed, color=OBS_C, alpha=0.6, lw=0, label='observed')
    if log:
        ax.set_yscale('log')
    ax.set_xlabel('median wind speed M (m/s)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(SPEEDS)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7, ncol=2)


def letter(ax, k, text=''):
    ax.text(0.02, 0.96, f'({"abcdefghij"[k]}) {text}', transform=ax.transAxes, va='top', fontweight='bold')


MIN_DAYS = 10             # station rates on fewer valid days are left out of the manuscript figure
NAMES = {'Alexander': 'Alexander', 'Meander': 'Meander', 'NoMeander': 'No-Meander'}


SEA_BREEZE_START = 10      # local hour at which the up-estuary (onshore) half of the model wind cycle is placed
SLOTS = [('none', 0)] + [('const', m) for m in SPEEDS] + [('diurnal', m) for m in SPEEDS]
DIURNAL_SLOTS = [('diurnal', m) for m in SPEEDS]          # Fig. 4e, f: the diurnal-wind runs only
CYCLE_LINTHRESH = 0.2      # PSU, linear range of the symmetric-log axis of Fig. 4e, f
CYCLE_YMAX = 2.0           # PSU, axis limit of Fig. 4e, f
CYCLE_LINEAR_SHARE = 0.3   # share of the axis height taken by the linear range


def _linscale(share=CYCLE_LINEAR_SHARE, linthresh=CYCLE_LINTHRESH, ymax=CYCLE_YMAX):
    '''linscale of the symlog axis that gives the linear range the requested share of the height'''
    from scipy.optimize import brentq
    from matplotlib.scale import SymmetricalLogTransform

    def lin_share(ls):
        tr = SymmetricalLogTransform(10, linthresh, ls)
        y = tr.transform(np.array([linthresh, ymax]))
        return y[0] / y[1] - share
    return brentq(lin_share, 1e-3, 50)


CYCLE_LINSCALE = _linscale()


def model_cycle(name, sensor, st='A3', ci=False):
    '''mean daily cycle (anomaly, PSU) of a run, by local hour (model hour + SEA_BREEZE_START), stratified part;
    with ci=True also the half width of its 90% confidence interval across the days of the run'''
    f = CACHE / f'{name}.csv'
    if not f.exists():
        return None
    w = stratified(pd.read_csv(f), st)
    if len(w) < 72:
        return None
    v = w[f'{st}_{sensor}'].reset_index(drop=True)
    anom = v - v.rolling(25, center=True).mean()          # 24-h running mean: removes the decline of the pool
    hour = (np.round(w.day.to_numpy() * 24).astype(int) % 24)
    g = anom.groupby(hour)
    cyc = g.mean().where(g.count() >= 3)
    if cyc.notna().sum() < 20:
        return None
    cyc.index = (cyc.index + SEA_BREEZE_START) % 24
    if not ci:
        return cyc.sort_index()
    n = g.count()
    half = (stats.t.ppf(0.95, n - 1) * g.std() / np.sqrt(n)).where(n >= 3)
    half.index = (half.index + SEA_BREEZE_START) % 24
    return cyc.sort_index(), half.sort_index()


def cycle_strip(a, sensor, obs_cycle, slots=SLOTS, obs_ci=None):
    '''one day per experiment: model mean daily cycles (domains) and the observed dry-season cycle (repeated)'''
    hours = np.arange(24)
    for k, (wind, m) in enumerate(slots):
        x = 24 * k + hours
        a.plot(x, obs_cycle.reindex(hours).values, color=OBS_C, lw=5, alpha=0.9, zorder=1)
        if obs_ci is not None:
            o, h = obs_cycle.reindex(hours).values, obs_ci.reindex(hours).values
            a.fill_between(x, o - h, o + h, color=OBS_CI_C, alpha=0.35, lw=0, zorder=2)
        for domain, prefix in DOMAINS.items():
            name = f'{prefix}none' if wind == 'none' else f'{prefix}{wind}_{m}'
            res = model_cycle(name, sensor, ci=True)
            if res is not None:
                cyc, half = res
                mean, h = cyc.reindex(hours).values, half.reindex(hours).values
                a.fill_between(x, mean - h, mean + h, color=DC[domain], alpha=0.15, lw=0, zorder=2)
                a.plot(x, mean, color=DC[domain], lw=1.6, ls=LS.get(wind, ':'), zorder=3)
        if k:
            a.axvline(24 * k, color='0.5', lw=0.8)
    a.axhline(0, color='0.6', lw=0.6, zorder=0)
    a.set_xlim(0, 24 * len(slots))
    a.set_xticks([24 * k + 12 for k in range(len(slots))])
    a.set_xticklabels(['no\nwind' if w == 'none' else f'{m}' for w, m in slots])
    for lab, k0 in (('constant wind', 1), ('diurnal wind', 1 + len(SPEEDS))):
        if slots is not SLOTS:
            break
        a.text(24 * (k0 + len(SPEEDS) / 2), 0.97, lab, transform=a.get_xaxis_transform(), ha='center', va='top',
               fontsize=12)
    a.grid(alpha=0.3, axis='y')


def manuscript_figure(t, lo, hi, obs):
    '''Fig. 4: the wind-variability experiment on the three domains (closed mouth; salinity itself, no seawater
    enters). Positions by distance from the mouth.'''
    from matplotlib.lines import Line2D
    from analyse_runs import STATIONS
    plt.rcParams.update({'font.size': 13})
    # top row: bigger panels close together, one-line y labels, (a) and (b) on one y axis
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 0.8, 0.8], hspace=0.3)
    row = gs[0].subgridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.24)
    ab = row[0].subgridspec(1, 2, wspace=0.05)
    a0 = fig.add_subplot(ab[0])
    top = [a0, fig.add_subplot(ab[1], sharey=a0), fig.add_subplot(row[1]), fig.add_subplot(row[2])]
    t = t.copy()
    for st in ('A4', 'A3'):
        t.loc[t[f'{st}_days'] < MIN_DAYS, f'{st}_exchange'] = np.nan
    t['A3_halocline_range'] = 2 * t['A3_halocline_halfrange']
    dist = {st: f'{d:.0f} m' for st, d in STATIONS.items()}
    panels = [('A4_exchange', 'exchange rate (% day$^{-1}$)', dist['A4']),
              ('A3_exchange', None, dist['A3']),
              ('pool_rate', 'salt loss rate (% day$^{-1}$)', 'whole estuary'),
              ('A3_halocline_range', 'halocline daily range (m)', dist['A3'])]
    for k, (a, (col, label, where)) in enumerate(zip(top, panels)):
        for domain in DOMAINS:
            for w in ('const', 'diurnal'):
                d = t[(t.domain == domain) & (t.wind == w)]
                a.plot(d.M, d[col], marker='o', ms=5, lw=1.8, ls=LS[w], color=DC[domain])
            n = t[(t.domain == domain) & (t.wind == 'none')]
            if len(n) and np.isfinite(n[col].iloc[0]):
                a.axhline(n[col].iloc[0], color=DC[domain], ls=':', lw=2.8)
        if col == 'A4_exchange':
            a.axhspan(lo, hi, color=OBS_C, alpha=0.7, lw=0)
        if label:
            a.set_ylabel(label)
        else:
            a.tick_params(labelleft=False)
        a.set_xticks(SPEEDS)
        a.set_xlim(0.4, 8.6)
        a.set_xlabel('M (m s$^{-1}$)')
        a.grid(alpha=0.3)
        a.text(0.04, 0.97, f'({"abcd"[k]}) {where}', transform=a.transAxes, va='top', fontweight='bold', fontsize=14,
               bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
    cyc = pd.read_csv(OBS / 'salinity_daily_cycle.csv')
    cyc = cyc[(cyc.station == 'A3') & (cyc.season == 'dry')]
    for r, (sensor, letter) in enumerate((('surface', 'e'), ('bottom', 'f')), start=1):
        a = fig.add_subplot(gs[r])
        oc = cyc[cyc.sensor == sensor].set_index('hour')
        cycle_strip(a, sensor, oc.anomaly, DIURNAL_SLOTS, obs_ci=oc.ci90)   # steady winds give no daily cycle
        a.set_yscale('symlog', linthresh=CYCLE_LINTHRESH, linscale=CYCLE_LINSCALE)
        a.set_ylim(-CYCLE_YMAX, CYCLE_YMAX)
        for yl in (-CYCLE_LINTHRESH, CYCLE_LINTHRESH):              # where the axis turns logarithmic (review)
            a.axhline(yl, color='0.35', lw=0.9, ls=(0, (4, 3)), zorder=0)
        a.set_yticks([-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1])
        a.set_yticklabels(['−1', '−0.5', '−0.2', '−0.1', '0', '0.1', '0.2', '0.5', '1'])
        a.yaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(linthresh=CYCLE_LINTHRESH, base=10, subs=range(2, 10)))
        a.set_ylabel(f'{sensor} salinity, {dist["A3"]},\ndeviation from the daily mean (PSU)')
        a.text(0.008, 0.96, f'({letter})', transform=a.transAxes, va='top', fontweight='bold', fontsize=14)
        if r == 2:
            a.set_xlabel('one mean day per diurnal-wind run (median wind speed M, m s$^{-1}$)')
    handles = [Line2D([], [], color=DC[d], lw=3, label=NAMES[d]) for d in DOMAINS]
    handles += [Line2D([], [], color='k', ls='-', lw=2, label='constant wind'),
                Line2D([], [], color='k', ls='--', lw=2, label='diurnal wind'),
                Line2D([], [], color='k', ls=':', lw=2.8, label='no wind'),
                Line2D([], [], color=OBS_C, lw=7, label='observed')]
    fig.legend(handles=handles, loc='lower center', ncol=7, mode='expand', fontsize=15, frameon=False,
               bbox_to_anchor=(0.04, 0.0, 0.92, 0.04), handlelength=2.6)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.11)
    out = REPO / 'docs' / 'figures' / 'manuscript_plot4.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--experiment', default='wind_iw_bar',
                    help='run folder under ROMS_runs (wind_iw_bar: closed mouth; wind_iw: old mouth, superseded)')
    a = ap.parse_args()
    set_experiment(a.experiment)
    OUT.mkdir(parents=True, exist_ok=True)
    items = runs()
    with Pool(a.jobs, initializer=set_experiment, initargs=(a.experiment,)) as pool:
        rows = pool.map(work, items)
    for item, r in zip(items, rows):
        if r is None:
            print(f'{item[3]}: not finished, skipped')
    t = pd.DataFrame([r for r in rows if r is not None])
    t.to_csv(OUT / 'wind_iw.csv', index=False, float_format='%.4g')
    cols = ['run', 'pool_rate', 'A4_exchange_orig', 'A4_days_orig', 'A4_exchange', 'A3_exchange_orig', 'A3_days_orig',
            'A3_exchange', 'K_eff', 'salt_left', 'orig_salt_left', 'seawater_m3', 'seawater_m3_per_day',
            'A4_halocline_halfrange', 'A3_halocline_halfrange', 'nan']
    with pd.option_context('display.width', 250):
        print(t[cols].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    tgt, lo, hi = cm.target()
    obs = pd.read_csv(OBS / 'salinity_diurnal.csv')
    obs = obs[obs.season == 'dry'].set_index(['station', 'sensor'])
    manuscript_figure(t, lo, hi, obs)
    plt.rcParams.update({'font.size': 11})

    # mixing of the original pool, the resupply from the sea, and the apparent (sensor) rates
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    pairs(axes[0, 0], t, 'A4_exchange_orig', 'A4 exchange rate, original pool (%/day)', (lo, hi))
    pairs(axes[0, 1], t, 'A3_exchange_orig', 'A3 exchange rate, original pool (%/day)', (lo, hi), legend=False)
    pairs(axes[0, 2], t, 'orig_salt_left', 'original-pool salt, day 30 / day 0', legend=False)
    pairs(axes[1, 0], t, 'A4_exchange', 'A4 exchange rate, salinity (%/day)', (lo, hi), legend=False)
    pairs(axes[1, 1], t, 'seawater_m3_per_day', 'net seawater entry (m³/day)', legend=False)
    pairs(axes[1, 2], t, 'pool_rate', 'loss rate of the original pool, whole estuary (%/day)', legend=False)
    for k, a in enumerate(axes.flat):
        letter(a, k)
    fig.tight_layout()
    fig.savefig(OUT / 'wind_iw_mixing.png', dpi=130)
    plt.close(fig)

    # internal motion: halocline and sensor daily cycles
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    pairs(axes[0, 0], t, 'A4_halocline_halfrange', 'A4 halocline depth, daily half range (m)')
    pairs(axes[1, 0], t, 'A3_halocline_halfrange', 'A3 halocline depth, daily half range (m)', legend=False)
    for r, st in enumerate(('A4', 'A3')):
        for c, sensor in enumerate(('surface', 'bottom'), start=1):
            v = obs.loc[(st, sensor), 'daily_half_range']
            pairs(axes[r, c], t, f'{st}_{sensor}_halfrange', f'{st} {sensor} salinity, daily half range (PSU)',
                  (0.95 * v, 1.05 * v), log=True, legend=False)
    for k, a in enumerate(axes.flat):
        letter(a, k)
    fig.tight_layout()
    fig.savefig(OUT / 'wind_iw_internal.png', dpi=130)
    plt.close(fig)

    # time series at M = 4: A4 bottom salinity (S and S_orig) and seawater volume
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
    for r, (domain, prefix) in enumerate(DOMAINS.items()):
        for k in ('const', 'diurnal', 'none'):
            name = f'{prefix}{k}_4' if k != 'none' else f'{prefix}none'
            if not (CACHE / f'{name}.csv').exists():
                continue
            s = pd.read_csv(CACHE / f'{name}.csv')
            ls = LS.get(k, ':')
            col = {'const': '#1f5fa8', 'diurnal': '#e08a1e', 'none': '#5c5c5c'}[k]
            axes[r, 0].plot(s.day, s.A4_bottom, color=col, ls='-', lw=1.0, label=f'{k}, S')
            axes[r, 0].plot(s.day, s.A4_orig_bottom, color=col, ls='--', lw=1.0, label=f'{k}, original pool')
            axes[r, 1].plot(s.day, s.seawater_m3 / 1e3, color=col, lw=1.2, label=k)
        axes[r, 0].set_ylabel(f'{domain}: A4 bottom salinity')
        axes[r, 1].set_ylabel(f'{domain}: seawater in estuary (10³ m³)')
        letter(axes[r, 0], 2 * r, f'{domain}, M = 4 m/s')
        letter(axes[r, 1], 2 * r + 1)
    for a in axes.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc='lower left')
    for a in axes[-1]:
        a.set_xlabel('day')
    fig.tight_layout()
    fig.savefig(OUT / 'wind_iw_series.png', dpi=130)
    plt.close(fig)

    # when does seawater enter? mean daily cycle of the net seawater flux in the diurnal runs
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    hours = np.arange(24)
    for a, (domain, prefix) in zip(axes, DOMAINS.items()):
        for m, c in zip(SPEEDS, plt.cm.viridis(np.linspace(0, 0.9, len(SPEEDS)))):
            f = CACHE / f'{prefix}diurnal_{m}.csv'
            if not f.exists():
                continue
            s = pd.read_csv(f)
            s = s[(s.day >= FIRST) & (s.day < LAST)]
            q = np.gradient(s.seawater_m3.values, s.day.values) / 24.0          # m3/h
            h = np.round((s.day.values % 1) * 24).astype(int) % 24
            a.plot(hours, pd.Series(q).groupby(h).mean().reindex(hours), color=c, marker='o', ms=3,
                   label=f'M = {m}')
        a.axvspan(0, 12, color='0.85', alpha=0.5, lw=0)
        a.axhline(0, color='0.5', lw=0.8)
        a.set_xticks(range(0, 25, 3))
        a.set_xlabel('model hour (shaded: onshore wind)')
        a.grid(alpha=0.3)
        letter(a, list(DOMAINS).index(domain), domain)
    axes[0].set_ylabel('net seawater entry (m³/h)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'wind_iw_seawater_cycle.png', dpi=130)
    plt.close(fig)

    # model spectra and daily cycle in the layout of salinity_spectra.py (diurnal wind, M = 4), with the
    # observed dry-season spectra in grey
    obs_sp = pd.read_csv(OBS / 'salinity_spectra.csv')
    obs_sp = obs_sp[obs_sp.season == 'dry']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for r, st in enumerate(('A4', 'A3')):
        for c, (domain, prefix) in enumerate(DOMAINS.items()):
            a = axes[r, c]
            f = CACHE / f'{prefix}diurnal_4.csv'
            for sensor in ('surface', 'bottom'):
                o = obs_sp[(obs_sp.station == st) & (obs_sp.sensor == sensor)]
                a.loglog(o.freq_cpd, o.psd, color=SENSOR_C[sensor], lw=0.8, alpha=0.35,
                         label=f'{sensor}, observed')
                if f.exists():
                    s = pd.read_csv(f)
                    fq, p = spectrum(s.day, s[f'{st}_{sensor}'])
                    a.loglog(fq[1:], p[1:], color=SENSOR_C[sensor], lw=1.4, label=f'{sensor}, model')
            for k in (1, 2, 3):
                a.axvline(k, color='0.6', lw=0.8, ls='--')
            a.set_xlim(1 / 8, 12)
            a.set_xlabel('frequency (cycles/day)')
            a.set_ylabel('PSD (PSU² per cycle/day)')
            a.grid(alpha=0.3, which='both')
            letter(a, 3 * r + c, f'{st}, {domain}')
            if r == 0 and c == 0:
                a.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(OUT / 'wind_iw_spectra.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
