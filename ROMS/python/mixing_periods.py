"""
Find the "fresh water only" periods in the Alexander salinity records, for the calibration of
vertical mixing.

Working assumption: during a prolonged period (>= 10 days) in which the bottom salinity at A4
falls at a steady rate, no seawater enters the estuary; the river is the only inflow, and the
decline is caused by vertical mixing of salt into the outflowing surface layer.

Data (ROMS_runs/mixing_calibration/input/, surface and bottom sensors ~0.2 m from the surface and
the bed, every 5-10 min):
    A4_salinity_2014-2025.csv   Michmoret Bridge, ~0.5 km from the mouth
    A3_salinity_2016-2025.csv   Turtle Bridge, ~3.5 km from the mouth

Method
    1. Despiking: a running median over 5 samples removes single-sample spikes (sensor noise:
       1.5% of the raw 5-10 min steps of the bottom series exceed 0.2 PSU, most of them
       isolated up-down pairs).
    2. Daily means, kept where the day has data in >= 80% of its hours.
    3. Periods at A4, searched greedily: from each start day the longest window (>= MIN_DAYS)
       that passes all criteria; the next search starts after its end. Criteria (defaults):
         steady decline   least-squares slope of the daily bottom salinity between
                          SLOPE_MIN and SLOPE_MAX (-1.0 ... -0.1 PSU/d), R^2 >= R2_MIN (0.9)
         no seawater      no day-to-day rise of the daily bottom salinity > DAY_RISE (0.3) and no
                          rise between consecutive despiked samples > STEP_RISE (0.2 PSU)
         fresh surface    surface (despiked) never more than SURF_SPIKE (3) above its period
                          median, and bottom - surface >= DS_MIN (3) on the first day
         coverage         valid days >= COVERAGE (90%) of the period
    4. A3 in the same period (if data): slope, R^2, coverage, and whether A3 shows a seawater
       pulse (same rise tests). A3 is not a selection criterion; it is reported.
    5. Exchange rate of each period, at A4 and A3. A slope in PSU/day depends on how much salt the
       pool holds above the surface water at the start, so it is not a measure of mixing. The
       measure is the fraction of the salt pool exchanged with the surface layer per day: a box
       model of the pool, V_b dS_b/dt = -Q (S_b - S_s), gives
           r = Q / V_b = -(dS_b/dt) / (S_b - S_s)        [% per day]
       with dS_b/dt the least-squares slope of the daily bottom salinity and S_b - S_s the period
       mean of the daily bottom - surface difference (days with both sensors). The 95% interval
       comes from that of the slope.
    6. Daily exchange rates (the calibration target): every period is cut into calendar days, and
       for each day r_d = -(slope of the despiked bottom samples of that day) / (mean bottom -
       surface of that day). Days with data in < DAY_HOURS of their hours are skipped. The mean
       of all days is the target exchange rate of the calibration. Writes
       input/daily_exchange.csv and, with --plot, daily_exchange_hist.png.

Writes input/periods.csv and, with --plot, figures in
ROMS_runs/mixing_calibration/output_figures/periods/.

Usage: python mixing_periods.py [--plot] [--min-days 10] [--slope-min -1.0] [--slope-max -0.1] ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import REPO

EXP = REPO.parent / 'ROMS_runs' / 'mixing_calibration'
INPUT = EXP / 'input'
FIGURES = EXP / 'output_figures' / 'periods'
FILES = {'A4': 'A4_salinity_2014-2025.csv', 'A3': 'A3_salinity_2016-2025.csv'}
DISTANCE_KM = {'A4': 0.5, 'A3': 3.5}

DEFAULTS = dict(min_days=10, slope_min=-1.0, slope_max=-0.1, r2_min=0.9, day_rise=0.3, step_rise=0.2,
                surf_spike=3.0, ds_min=3.0, coverage=0.9, day_hours=0.8, despike=5)

# colours: bottom / surface per station (fixed order); season shading
C_BOT = {'A4': '#1f5fa8', 'A3': '#2a9d4b'}
C_SURF = {'A4': '#7fb0e0', 'A3': '#9ad3a6'}
C_FIT = '#b2392f'
SEASON_FILL = {'dry': '#f2c46b', 'wet': '#9cb8d9'}


def load_station(name):
    d = pd.read_csv(INPUT / FILES[name], parse_dates=[0])
    d.columns = ['time', 'surface', 'bottom']
    return d.set_index('time').sort_index()


def despike(d, n):
    return d.rolling(n, center=True, min_periods=1).median()


def daily(d, min_hours_frac):
    hours = d.notna().groupby(d.index.floor('h')).any().groupby(lambda t: t.floor('D')).sum()
    mean = d.resample('D').mean()
    hours = hours.reindex(mean.index).fillna(0)
    for c in mean:
        mean.loc[hours[c] < 24 * min_hours_frac, c] = np.nan
    return mean


def fit(series):
    '''least-squares slope (per day), its 95% half-width and R^2 of a daily series with gaps.'''
    y = series.dropna()
    if len(y) < 3:
        return np.nan, np.nan, np.nan
    x = (y.index - series.index[0]).days.to_numpy(float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se = np.sqrt(ss_res / max(len(y) - 2, 1) / ((x - x.mean()) ** 2).sum())
    return coef[0], 1.96 * se, r2


def rises(day_bottom, raw_bottom):
    '''largest day-to-day rise of the daily bottom salinity and largest rise between samples.'''
    d = day_bottom.dropna()
    day = float(np.diff(d.to_numpy()).max()) if len(d) > 1 else np.nan
    r = raw_bottom.dropna()
    step = float(np.diff(r.to_numpy()).max()) if len(r) > 1 else np.nan
    return day, step


def check(p, day, raw, s, e):
    '''criteria for the window [s, e] (days, inclusive); returns (ok, numbers).'''
    w = day.loc[s:e]
    n_days = (e - s).days + 1
    cov = w.bottom.notna().sum() / n_days
    slope, ci, r2 = fit(w.bottom)
    rw = raw.loc[s:e + pd.Timedelta('1D') - pd.Timedelta('1s')]
    day_rise, step_rise = rises(w.bottom, rw.bottom)
    surf = rw.surface.dropna()
    spike = float(surf.max() - surf.median()) if len(surf) else np.nan
    first = w.dropna().iloc[0] if w.dropna().size else None
    ds0 = float(first.bottom - first.surface) if first is not None else np.nan
    ok = (cov >= p['coverage'] and p['slope_min'] <= slope <= p['slope_max'] and r2 >= p['r2_min']
          and day_rise <= p['day_rise'] and step_rise <= p['step_rise']
          and (np.isnan(spike) or spike <= p['surf_spike']) and ds0 >= p['ds_min'])
    return ok, dict(days=n_days, coverage=cov, slope=slope, slope_ci95=ci, r2=r2, max_day_rise=day_rise,
                    max_step_rise=step_rise, surface_spike=spike, dS_start=ds0)


def find_periods(p, day, raw):
    days = day.index
    periods, i = [], 0
    L = p['min_days']
    while i + L <= len(days):
        s = days[i]
        ok, _ = check(p, day, raw, s, days[i + L - 1])
        if not ok:
            i += 1
            continue
        j = i + L - 1
        while j + 1 < len(days) and check(p, day, raw, s, days[j + 1])[0]:
            j += 1
        periods.append((s, days[j]))
        i = j + 1
    return periods


def station_stats(p, day, raw, s, e, prefix):
    w = day.loc[s:e]
    n_days = (e - s).days + 1
    slope, ci, r2 = fit(w.bottom)
    rw = raw.loc[s:e + pd.Timedelta('1D') - pd.Timedelta('1s')]
    day_rise, step_rise = rises(w.bottom, rw.bottom)
    valid = w.bottom.dropna()
    out = {f'{prefix}_coverage': w.bottom.notna().sum() / n_days,
           f'{prefix}_bottom_start': valid.iloc[0] if len(valid) else np.nan,
           f'{prefix}_bottom_end': valid.iloc[-1] if len(valid) else np.nan,
           f'{prefix}_surface_mean': w.surface.mean(),
           f'{prefix}_slope': slope, f'{prefix}_slope_ci95': ci, f'{prefix}_r2': r2,
           f'{prefix}_max_day_rise': day_rise, f'{prefix}_max_step_rise': step_rise}
    out.update({f'{prefix}_{k}': v for k, v in exchange(w, slope, ci).items()})
    return out


def exchange(w, slope, ci):
    '''exchange rate of the salt pool (% per day) and its 95% half-width; see the module docstring.'''
    ds = (w.bottom - w.surface).dropna()
    ds_mean = float(ds.mean()) if len(ds) else np.nan
    if not ds_mean > 0:
        return dict(dS_mean=ds_mean, exchange=np.nan, exchange_ci95=np.nan)
    return dict(dS_mean=ds_mean, exchange=-100 * slope / ds_mean, exchange_ci95=100 * ci / ds_mean)


def build_table(p):
    raw, day = {}, {}
    for name in FILES:
        raw[name] = despike(load_station(name), p['despike'])
        day[name] = daily(raw[name], p['day_hours'])
    rows = []
    for k, (s, e) in enumerate(find_periods(p, day['A4'], raw['A4']), start=1):
        row = {'id': f'P{k:02d}', 'start': s.date(), 'end': e.date(), 'days': (e - s).days + 1,
               'season': 'dry' if 5 <= (s + (e - s) / 2).month <= 10 else 'wet'}
        _, num = check(p, day['A4'], raw['A4'], s, e)
        row.update(station_stats(p, day['A4'], raw['A4'], s, e, 'A4'))
        row['A4_dS_start'] = num['dS_start']
        row['A4_surface_spike'] = num['surface_spike']
        if s >= day['A3'].index[0]:
            st = station_stats(p, day['A3'], raw['A3'], s, e, 'A3')
            if st['A3_coverage'] > 0:
                st['A3_pulse'] = bool(st['A3_max_day_rise'] > p['day_rise'])      # daily rise only: the A3 record is noisier
                row.update(st)
        rows.append(row)
    return pd.DataFrame(rows), raw, day


def daily_exchange(table, raw, day, station='A4'):
    '''exchange rate (% per day) of each calendar day of each period; see step 6 of the docstring.'''
    rows = []
    for _, r in table.iterrows():
        for d in pd.date_range(r.start, r.end, freq='D'):
            if pd.isna(day[station].bottom.get(d, np.nan)):
                continue
            w = raw[station].loc[d:d + pd.Timedelta('1D') - pd.Timedelta('1s')].dropna()
            if len(w) < 3:
                continue
            x = (w.index - d).total_seconds().to_numpy() / 86400
            slope = np.polyfit(x, w.bottom.to_numpy(), 1)[0]
            ds = float((w.bottom - w.surface).mean())
            rows.append({'id': r.id, 'season': r.season, 'date': d.date(), 'slope': slope, 'dS': ds,
                         'exchange': -100 * slope / ds if ds > 0 else np.nan})
    return pd.DataFrame(rows)


def exchange_histogram(daily_ex, out):
    '''histogram of the daily exchange rates, all periods and dry-season periods, with their means.'''
    fig, ax = plt.subplots(figsize=(8, 4.2))
    v = daily_ex.exchange.dropna()
    bins = np.arange(np.floor(v.min() / 2) * 2, np.ceil(v.max() / 2) * 2 + 2, 2)
    dry = daily_ex[daily_ex.season == 'dry'].exchange.dropna()
    ax.hist(v, bins=bins, color='#9cb8d9', edgecolor='white', label=f'all periods ({len(v)} days)')
    ax.hist(dry, bins=bins, color='#c98a12', edgecolor='white', alpha=0.8, label=f'dry season ({len(dry)} days)')
    ax.axvline(v.mean(), color='#1f5fa8', lw=2, label=f'mean, all: {v.mean():.1f} %/d')
    ax.axvline(dry.mean(), color='#8a5a00', lw=2, ls='--', label=f'mean, dry: {dry.mean():.1f} %/d')
    ax.axvline(0, color='0.3', lw=0.8)
    ax.set_xlabel('daily exchange rate at A4 (% of the salt pool per day)')
    ax.set_ylabel('days')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'daily_exchange_hist.png', dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- figures

def overview_figure(table, day, out):
    years = sorted(set(day['A4'].index.year))
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 1.55 * len(years)), sharey=True)
    for ax, y in zip(axes, years):
        for name in ('A4', 'A3'):
            d = day[name].loc[f'{y}-01-01':f'{y}-12-31']
            ax.plot(d.index, d.bottom, color=C_BOT[name], lw=1.1)
            ax.plot(d.index, d.surface, color=C_SURF[name], lw=0.8)
        for _, r in table.iterrows():
            s, e = pd.Timestamp(r.start), pd.Timestamp(r.end) + pd.Timedelta('1D')
            if s.year == y or e.year == y:
                ax.axvspan(s, e, color=SEASON_FILL[r.season], alpha=0.55, lw=0)
                ax.text(s + (e - s) / 2, 39, r.id, fontsize=6, ha='center', va='top')
        ax.set_xlim(pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y + 1}-01-01'))
        ax.set_ylim(0, 40)
        ax.text(0.003, 0.92, str(y), transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[len(years) // 2].set_ylabel('salinity (daily mean)')
    handles = [plt.Line2D([], [], color=C_BOT['A4'], lw=1.5, label='A4 bottom'),
               plt.Line2D([], [], color=C_SURF['A4'], lw=1.5, label='A4 surface'),
               plt.Line2D([], [], color=C_BOT['A3'], lw=1.5, label='A3 bottom'),
               plt.Line2D([], [], color=C_SURF['A3'], lw=1.5, label='A3 surface'),
               plt.Rectangle((0, 0), 1, 1, color=SEASON_FILL['dry'], alpha=0.55, label='period, dry season (May–Oct)'),
               plt.Rectangle((0, 0), 1, 1, color=SEASON_FILL['wet'], alpha=0.55, label='period, wet season (Nov–Apr)')]
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / 'overview.png', dpi=130)
    plt.close(fig)


def draw_period(ax, r, raw, day, pad_days=3, compact=False):
    s, e = pd.Timestamp(r.start), pd.Timestamp(r.end) + pd.Timedelta('1D')
    lo, hi = s - pd.Timedelta(days=pad_days), e + pd.Timedelta(days=pad_days)
    for name in ('A4', 'A3'):
        w = raw[name].loc[lo:hi]
        if w.empty:
            continue
        ax.plot(w.index, w.bottom, color=C_BOT[name], lw=0.6, alpha=0.8)
        ax.plot(w.index, w.surface, color=C_SURF[name], lw=0.6, alpha=0.9)
        d = day[name].loc[s:e - pd.Timedelta('1D')]
        ax.plot(d.index + pd.Timedelta('12h'), d.bottom, 'o', color=C_BOT[name], ms=2.5 if compact else 3.5)
    d = day['A4'].loc[s:e - pd.Timedelta('1D')].bottom
    x = (d.index - s).days.to_numpy(float)
    ok = d.notna().to_numpy()
    coef = np.polyfit(x[ok], d.to_numpy()[ok], 1)
    ax.plot(d.index + pd.Timedelta('12h'), np.polyval(coef, x), color=C_FIT, lw=1.6)
    ax.axvspan(s, e, color=SEASON_FILL[r.season], alpha=0.25, lw=0)
    ax.set_ylim(0, 40)
    txt = f'{r.id}  {r.start} – {r.end} ({r.days} d, {r.season})\nA4 exchange {r.A4_exchange:.1f} ± {r.A4_exchange_ci95:.1f} %/d (bottom {r.A4_slope:+.2f} PSU/d, R² {r.A4_r2:.2f})'
    if not compact and pd.notna(r.get('A3_slope', np.nan)):
        txt += (f'\nA3 exchange {r.A3_exchange:.1f} %/d (bottom {r.A3_slope:+.2f} PSU/d, R² {r.A3_r2:.2f})'
                + ('  (A3 pulse)' if r.A3_pulse else ''))
    ax.set_title(txt, fontsize=6.5 if compact else 9, loc='left')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5 if compact else 8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b' if compact else '%d %b %Y'))
    ax.tick_params(labelsize=6 if compact else 8)
    ax.grid(alpha=0.25)


def legend_handles():
    return [plt.Line2D([], [], color=C_BOT['A4'], lw=1.5, label='A4 bottom'),
            plt.Line2D([], [], color=C_SURF['A4'], lw=1.5, label='A4 surface'),
            plt.Line2D([], [], color=C_BOT['A3'], lw=1.5, label='A3 bottom'),
            plt.Line2D([], [], color=C_SURF['A3'], lw=1.5, label='A3 surface'),
            plt.Line2D([], [], color=C_FIT, lw=1.6, label='linear fit, A4 bottom (daily means = dots)')]


def period_figures(table, raw, day, out):
    for _, r in table.iterrows():
        fig, ax = plt.subplots(figsize=(9, 4))
        draw_period(ax, r, raw, day)
        ax.set_ylabel('salinity')
        ax.legend(handles=legend_handles(), fontsize=7, loc='upper right', ncol=2)
        fig.tight_layout()
        fig.savefig(out / f'period_{r.id}.png', dpi=120)
        plt.close(fig)
    n = len(table)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.5 * rows), sharey=True, squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), table.iterrows()):
        draw_period(ax, r, raw, day, pad_days=2, compact=True)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / 'periods_grid.png', dpi=110)
    plt.close(fig)


def slope_summary(table, out):
    '''decline rate (PSU/day, top) and exchange rate (%/day, bottom) against the initial stratification
    and the month; the top row depends on the initial condition, the bottom row does not.'''
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row')
    for (a1, a2), col, ci, label in ((axes[0], 'A4_slope', 'A4_slope_ci95', 'A4 bottom salinity trend (PSU/day)'),
                                     (axes[1], 'A4_exchange', 'A4_exchange_ci95', 'A4 exchange rate (% of pool/day)')):
        for season, m in (('dry', 'o'), ('wet', 's')):
            t = table[table.season == season]
            c = '#c98a12' if season == 'dry' else '#3d6fa8'
            a1.errorbar(t.A4_dS_start, t[col], yerr=t[ci], fmt=m, color=c, ms=5, lw=0.8,
                        label=f'{season} season ({len(t)})')
            months = pd.to_datetime(t.start).dt.month + pd.to_datetime(t.start).dt.day / 31
            a2.plot(months, t[col], m, color=c, ms=5)
        a1.set_ylabel(label)
        a2.set_xticks(range(1, 13))
        a2.set_xticklabels(list('JFMAMJJASOND'))
        for a in (a1, a2):
            a.grid(alpha=0.3)
    axes[1, 0].set_xlabel('bottom − surface salinity at A4, first day')
    axes[1, 1].set_xlabel('month of the start')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'slope_summary.png', dpi=120)
    plt.close(fig)


def main(p, plot=False):
    table, raw, day = build_table(p)
    table.to_csv(INPUT / 'periods.csv', index=False, float_format='%.4g')
    cols = ['id', 'start', 'end', 'days', 'season', 'A4_bottom_start', 'A4_bottom_end', 'A4_surface_mean',
            'A4_slope', 'A4_r2', 'A4_dS_mean', 'A4_exchange', 'A4_exchange_ci95', 'A3_slope', 'A3_r2',
            'A3_exchange', 'A3_pulse']
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table[[c for c in cols if c in table]].round(2).to_string(index=False))
    print(f'{len(table)} periods ({(table.season == "dry").sum()} dry season); written to {INPUT / "periods.csv"}')
    daily_ex = daily_exchange(table, raw, day)
    daily_ex.to_csv(INPUT / 'daily_exchange.csv', index=False, float_format='%.4g')
    for name, v in (('all', daily_ex.exchange), ('dry', daily_ex[daily_ex.season == 'dry'].exchange)):
        v = v.dropna()
        print(f'daily exchange rate, {name}: {len(v)} days, mean {v.mean():.2f} %/d (95% CI '
              f'±{1.96 * v.std() / np.sqrt(len(v)):.2f}), median {v.median():.2f}, sd {v.std():.2f}, '
              f'{(v < 0).mean() * 100:.0f}% of days negative')
    if plot:
        FIGURES.mkdir(parents=True, exist_ok=True)
        exchange_histogram(daily_ex, FIGURES)
        FIGURES.mkdir(parents=True, exist_ok=True)
        overview_figure(table, day, FIGURES)
        period_figures(table, raw, day, FIGURES)
        slope_summary(table, FIGURES)
        print(f'figures in {FIGURES}')
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Find steady bottom-salinity decline periods (A4, A3)')
    ap.add_argument('--plot', action='store_true')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = vars(ap.parse_args())
    plot = a.pop('plot')
    main(a, plot)
