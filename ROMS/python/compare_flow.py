"""
E3: river-flow experiment, on the three domains. Compares the runs of
ROMS_runs/<experiment> (flow_bar: the closed mouth): steady river flow x0.25 ... x16 of Q_REF = 20 000 m3/day
(<prefix>q025 ... <prefix>q16) and no river (<prefix>q0, the floor), no wind, sea level 0, 30 days, hourly
history. Prefixes: alex_ (Alexander), meander_ (Meander), nomeander_ (NoMeander). A decay experiment: every run
starts from the same salt pool (the filled estuary), and the question is how fast each flow removes it.

Per run (section 5.4 of the plan; ADJUST = first 0.25 day left out of the fits, while the level over the crest
adjusts to the new flow):
    5.4.1  pool salt     salt deeper than POOL_DEPTH below the initial mean surface (the observed halocline
                         depth; a fixed level for the whole run, so that the rise of the level at high flow
                         does not move it);
                         P(t) = pool salt / its initial value; T50, T90 (first time P < 0.5, 0.1), e-folding time
                         and pool rate r_pool = 100 (1 - exp(-k)) %/day, k the slope of -ln P from ADJUST until
                         P < POOL_END; t* = Q t / V flushing volumes (V the initial estuary volume)
    5.4.2  exchange      sensor exchange rate at A4 and A3 (0.2 m below the surface / above the bed), as in the
                         wind-variability experiment: daily rates (calibrate_mixing.daily_exchange) on days 2-30
                         with S_b - S_s >= DAILY_DS_MIN, averaged when there are at least DAILY_MIN_DAYS such days.
                         Where the pool goes faster than that (high flows), 24-h regressions stepped by 1 h from
                         ADJUST, averaged over windows with S_b - S_s > DS_MIN (method 'running' in flow.csv)
    5.4.4  regime        over the first REGIME_DAYS after ADJUST: per cross-section, two layers split at the
                         gradient-weighted halocline; surface-layer speed u1 and thickness h1, g' = g beta dS;
                         densimetric Froude number Fr = u1 / sqrt(g' h1) in the columns with dS > DS_MIN,
                         cross-section and time mean; its maximum along the channel
                         and where. Share of the halocline cells (dS/dz > 1 PSU/m) with gradient Richardson number
                         below 0.25, and their mean AKs
    5.4.5  mixing        salinity-square budget of the estuary cells (Burchard and Rennau 2008):
                             d/dt int S^2 dV = - F_out - chi_phys - chi_num,
                         F_out the S^2 flux out over the crest face (upwind S), chi_phys = int 2 AKs (dS/dz)^2 dV
                         (hourly snapshots of AKs and S), chi_num the residual (numerical mixing of advection);
                         integrated from 0 until P < POOL_END (or the end); numerical share chi_num / (sum)
    5.4.6  where         upstream end of the pool (farthest centreline cell with bed S >= S_POOL_END) against
                         time, its retreat speed; salt left per REACH_M reach of the channel at t* = 0.5, 1, 2;
                         centreline salinity sections at t* = SECTION_TSTAR
    5.4.7  ages          ventilation age (dye_03) of the bottom water at A4 and A3 and of the pool, river-water
                         fraction (dye_01) of the pool; time mean after ADJUST and end value
Across runs:
    5.4.3  scaling       log r against log Q (Q > 0), slope n with its 95% interval, per domain, for r_pool, A4
                         and A3; Q_c, the lowest flow with T90 <= TC_DAYS (log interpolation)
    5.4.8  shape         Meander/NoMeander and Alexander/NoMeander of r_pool, T50, A4 rate and the numerical share
5.4.9 (observed flows) waits for a daily flow record.

The per-run results are cached in output_figures/cache/<run>.pkl (rebuilt when roms_his.nc is newer).

Writes ROMS_runs/<experiment>/output_figures/: flow.csv (one row per run), flow_scaling.csv, flow_pool.png,
flow_rates.png, flow_mixing.png, flow_shape.png, flow_where.png, flow_sections_<run suffix>.png, and the
manuscript Fig. 5, docs/figures/manuscript_plot5.png.

Usage: python compare_flow.py [--experiment flow_bar] [--jobs 6]
"""
import argparse
import pickle
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import calibrate_mixing as cm
from analyse_runs import RUNS_DIR, run_series, station_cells
from common import estuary_slice, mouth_index
from compare_layers import depths

Q_REF = 20000.0            # m3/day, river flow at RIVER_SCALE = 1
SCALES = {0: 'q0', 0.25: 'q025', 0.5: 'q05', 1: 'q1', 2: 'q2', 4: 'q4', 8: 'q8', 16: 'q16', 32: 'q32', 64: 'q64',
          128: 'q128'}
DOMAINS = {'Alexander': 'alex_', 'Meander': 'meander_', 'NoMeander': 'nomeander_'}
POOL_DEPTH = 0.7           # m, the pool is the water deeper than this below the initial surface (observed
                           # halocline depth), a fixed level for the whole run (~ -0.55 m)
ADJUST = 0.25              # days left out of the fits
POOL_END = 0.05            # fits and budgets stop when this fraction of the pool salt is left
DS_MIN = 1.0               # PSU, smallest bottom - surface difference for a running (24-h window) exchange rate
DAILY_DS_MIN = 3.0         # PSU, the same for the daily rates (as compare_wind_iw.DS_MIN)
DAILY_FIRST = 1            # daily rates from day 2 (calibrate_mixing.daily_exchange labels a day by its end)
DAILY_MIN_DAYS = 10        # daily rates are used when at least this many days qualify (as compare_wind_iw)
REGIME_DAYS = 2.0          # days over which the regime numbers are averaged
S_POOL_END = 5.0           # bed isohaline that marks the upstream end of the pool
REACH_M = 500.0            # length of the reaches for the salt left along the channel
TSTAR_REACH = (0.5, 1, 2)
SECTION_TSTAR = (0.25, 0.5, 1, 2)
SECTION_RUNS = ('q1', 'q4', 'q16')
TC_DAYS = 5.0              # Q_c: lowest flow that removes 90% of the pool within this time
G, BETA, RHO0 = 9.81, 7.7e-4, 1025.0
OBS_C, OBS_CI_C = '#945099', '#5e2f63'     # observed: purple, unlike any domain colour
DC = {'Alexander': '#1f5fa8', 'Meander': '#e08a1e', 'NoMeander': '#2a9d4b'}
NAMES = {'Alexander': 'Alexander', 'Meander': 'Meander', 'NoMeander': 'No-Meander'}


def set_experiment(name):
    '''point EXP, OUT and CACHE at ROMS_runs/<name> (also in the Pool workers, which do not share globals).'''
    global EXP, OUT, CACHE
    EXP = RUNS_DIR / name
    OUT = EXP / 'output_figures'
    CACHE = OUT / 'cache'


set_experiment('flow_bar')


def finished(path):
    log = path / 'roms.log'
    return (path / 'roms_his.nc').exists() and log.exists() and 'ROMS: DONE' in log.read_text(errors='replace')


def along(grid):
    '''centre row, and the distance (m) from the mouth along the centreline of every xi index from the mouth on'''
    j = grid.sizes['eta_rho'] // 2
    m = mouth_index(grid)
    x, y = grid.x_rho.values[j, m:], grid.y_rho.values[j, m:]
    return j, np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def to_rho(u, v):
    '''u and v (N, J, I-1), (N, J-1, I) averaged to the rho points, zero on the edges'''
    ur = np.zeros(u.shape[:2] + (u.shape[2] + 1,))
    vr = np.zeros(v.shape[:1] + (v.shape[1] + 1,) + v.shape[2:])
    ur[..., 1:-1] = 0.5 * (u[..., :-1] + u[..., 1:])
    vr[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    return ur, vr


def regime(s, rho, zr, zw, u, v, aks, sl):
    '''per xi index of the estuary: cross-section mean Fr; and the halocline cells with Ri < 0.25'''
    dz = np.diff(zw, axis=0)
    g = np.maximum(-np.diff(s, axis=0) / np.diff(zr, axis=0), 0.0)
    zm = 0.5 * (zr[1:] + zr[:-1])
    zh = (zm * g).sum(0) / np.maximum(g.sum(0), 1e-12)
    up = zr > zh
    ur, vr = to_rho(u, v)
    h1 = (dz * up).sum(0)
    h2 = (dz * ~up).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        u1 = np.hypot((ur * dz * up).sum(0), (vr * dz * up).sum(0)) / h1
        ds = (s * dz * ~up).sum(0) / h2 - (s * dz * up).sum(0) / h1
        fr = u1 / np.sqrt(G * BETA * np.maximum(ds, 1e-3) * h1)
    fr = np.where((h1 > 0) & (h2 > 0) & (ds > DS_MIN), fr, np.nan)[sl]       # stratified columns only
    # gradient Richardson number at the interior w levels
    dzr = np.diff(zr, axis=0)
    n2 = -G / RHO0 * np.diff(rho, axis=0) / dzr
    sh2 = (np.diff(ur, axis=0) / dzr) ** 2 + (np.diff(vr, axis=0) / dzr) ** 2
    sl3 = (slice(None),) + sl
    halo = (np.abs(np.diff(s, axis=0) / dzr) > 1.0)[sl3]
    ri = (n2 / np.maximum(sh2, 1e-12))[sl3]
    k = aks[1:-1][sl3]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # sections with no stratified column
        frm = np.nanmean(fr, axis=0)
    return frm, int(halo.sum()), int((halo & (ri < 0.25)).sum()), float(k[halo].sum())


def analyse_run(path, q):
    '''hourly series and per-run fields of one run (see the module docstring)'''
    ds = xr.open_dataset(path / 'roms_his.nc', decode_times=False)
    grid = xr.open_dataset(path / 'grid.nc')
    dA = 1.0 / (grid.pm.values * grid.pn.values)
    inner = estuary_slice(grid, 3)
    sl = estuary_slice(grid)
    m = mouth_index(grid)
    jc, dist = along(grid)
    jst, cells = station_cells(grid)
    rows_e = slice(1, -1)
    dy_u = 2.0 / (grid.pn.values[rows_e, m] + grid.pn.values[rows_e, m + 1])
    ncol = grid.sizes['xi_rho'] - 1 - (m + 1)          # estuary columns
    col_dist = dist[1:1 + ncol]
    reach = np.floor(col_dist / REACH_M).astype(int)
    rows, reach_salt, sections = [], [], []
    fr_sum, fr_n, halo_n, halo_lowri, halo_aks = np.zeros(ncol), np.zeros(ncol), 0, 0, 0.0
    days = (ds.ocean_time.values - ds.ocean_time.values[0]) / 86400.0
    z_pool = float(ds.zeta.isel(ocean_time=0).values[sl].mean()) - POOL_DEPTH     # fixed for the whole run
    for t in range(ds.sizes['ocean_time']):
        zw, zr = depths(ds, t), depths(ds, t, 'r')
        dz = np.diff(zw, axis=0)
        dV = dz * dA
        s = ds.salt.isel(ocean_time=t).values
        aks = ds.AKs.isel(ocean_time=t).values
        u = ds.u.isel(ocean_time=t).values
        pool = zr < z_pool
        row = {'pool_salt': float((s * dV * pool)[inner].sum()), 'pool_volume': float((dV * pool)[inner].sum()),
               's2': float((s ** 2 * dV)[inner].sum())}
        dsdz = np.diff(s, axis=0) / np.diff(zr, axis=0)
        row['chi_phys'] = float((2 * aks[1:-1] * dsdz ** 2 * np.diff(zr, axis=0) * dA)[inner].sum())
        # flux over the crest face (u index m, between rho m and m+1); negative u is towards the sea
        uf = u[:, rows_e, m]
        hz = 0.5 * (dz[:, rows_e, m] + dz[:, rows_e, m + 1])
        s_up = np.where(uf < 0, s[:, rows_e, m + 1], s[:, rows_e, m])
        row['face_flow'] = float((uf * hz * dy_u).sum())
        row['f_out_s2'] = float(-(uf * hz * dy_u * s_up ** 2).sum())
        if 'dye_03' in ds:
            c1 = ds.dye_01.isel(ocean_time=t).values
            c3 = ds.dye_03.isel(ocean_time=t).values
            a3 = ds.dye_03_age.isel(ocean_time=t).values
            w = (dV * pool)[inner]
            row['pool_age_h'] = float((a3[inner] * w).sum() / max((c3[inner] * w).sum(), 1e-12) / 3600)
            row['pool_river_fraction'] = float((c1[inner] * w).sum() / max(w.sum(), 1e-12))
            for name, i in cells.items():
                row[f'{name}_bottom_age_h'] = float(a3[0, jst, i] / max(c3[0, jst, i], 1e-12) / 3600)
        bed = s[0, jc, m + 1:m + 1 + ncol]
        above = np.nonzero(bed >= S_POOL_END)[0]
        row['pool_end_m'] = float(col_dist[above.max()]) if len(above) else 0.0
        rows.append(row)
        reach_salt.append(np.bincount(reach, weights=(s * dV)[inner].sum(axis=(0, 1)), minlength=reach.max() + 1))
        if ADJUST <= days[t] <= ADJUST + REGIME_DAYS:
            fr, nh, nl, ka = regime(s, ds.rho.isel(ocean_time=t).values, zr, zw, u,
                                    ds.v.isel(ocean_time=t).values, aks, sl)
            ok = np.isfinite(fr)
            fr_sum[ok] += fr[ok]
            fr_n[ok] += 1
            halo_n, halo_lowri, halo_aks = halo_n + nh, halo_lowri + nl, halo_aks + ka
        sections.append((s[:, jc, m + 1:m + 1 + ncol], zr[:, jc, m + 1:m + 1 + ncol]))
    df = pd.concat([run_series(path), pd.DataFrame(rows)], axis=1)
    volume = float(df.volume_m3.iloc[0])
    df['tstar'] = q * df.day / (volume / 86400.0) if q > 0 else 0.0
    keep = {}
    for ts in SECTION_TSTAR:
        if q > 0 and df.tstar.iloc[-1] >= ts:
            keep[ts] = sections[int(np.argmin(np.abs(df.tstar.values - ts)))]
    reach_salt = np.array(reach_salt)
    reach_left = {}
    for ts in TSTAR_REACH:
        if q > 0 and df.tstar.iloc[-1] >= ts:
            k = int(np.argmin(np.abs(df.tstar.values - ts)))
            reach_left[ts] = reach_salt[k] / np.maximum(reach_salt[0], 1e-9)
    with np.errstate(invalid='ignore'):
        fr_profile = fr_sum / fr_n
    return {'series': df, 'volume': volume, 'col_dist': col_dist, 'reach_left': reach_left,
            'reach_x': (np.arange(reach.max() + 1) + 0.5) * REACH_M, 'sections': keep, 'fr_profile': fr_profile,
            'halo': (halo_n, halo_lowri, halo_aks)}


def cached(path, q):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{path.name}.pkl'
    if f.exists() and f.stat().st_mtime > (path / 'roms_his.nc').stat().st_mtime:
        return pickle.loads(f.read_bytes())
    r = analyse_run(path, q)
    f.write_bytes(pickle.dumps(r))
    return r


def crossing(day, p, level):
    below = np.nonzero(p.values < level)[0]
    if not len(below):
        return np.nan
    k = below[0]
    if k == 0:
        return float(day.iloc[0])
    return float(np.interp(level, [p.iloc[k], p.iloc[k - 1]], [day.iloc[k], day.iloc[k - 1]]))


def running_exchange(s, st):
    '''mean exchange rate (%/day) of 24-h regressions stepped by 1 h, and the number of valid windows'''
    d = s.day.values
    b, top = s[f'{st}_bottom'].values, s[f'{st}_surface'].values
    rates = []
    for k in np.nonzero(d >= ADJUST)[0]:
        w = (d >= d[k]) & (d <= d[k] + 1)
        if w.sum() < 13 or d[w][-1] - d[k] < 0.99:
            break
        dS = (b[w] - top[w]).mean()
        if dS > DS_MIN:
            rates.append(-100 * np.polyfit(d[w], b[w], 1)[0] / dS)
    return (float(np.mean(rates)) if rates else np.nan), len(rates)


def daily_exchange(s, st):
    '''mean daily exchange rate (%/day) as in the wind-variability experiment, and the number of days'''
    cm.FIRST_DAY, cm.LAST_DAY = DAILY_FIRST, int(s.day.iloc[-1])
    d = cm.daily_exchange(s, st)
    d = d[d.dS >= DAILY_DS_MIN]
    return (float(d.exchange.mean()) if len(d) else np.nan), len(d)


def summarise(r, q):
    s = r['series']
    p = s.pool_salt / s.pool_salt.iloc[0]
    row = {'T50': crossing(s.day, p, 0.5), 'T90': crossing(s.day, p, 0.1), 'pool_left_end': float(p.iloc[-1])}
    end = s.day[p < POOL_END].min() if (p < POOL_END).any() else s.day.iloc[-1]
    w = s[(s.day >= ADJUST) & (s.day <= end)]
    if len(w) < 6:
        w = s[s.day >= ADJUST].iloc[:6]
    k = -np.polyfit(w.day, np.log(np.maximum(w.pool_salt / s.pool_salt.iloc[0], 1e-12)), 1)[0]
    row['pool_rate'] = 100 * (1 - np.exp(-k))
    row['efold_days'] = 1 / k if k > 0 else np.nan
    for st in ('A4', 'A3'):
        rate, n = daily_exchange(s, st)
        if n >= DAILY_MIN_DAYS:
            row[f'{st}_exchange'], row[f'{st}_days'], row[f'{st}_method'] = rate, n, 'daily'
        else:
            rate, n = running_exchange(s, st)
            row[f'{st}_exchange'], row[f'{st}_days'], row[f'{st}_method'] = rate, n / 24.0, 'running'
    # salinity-square budget, 0 .. end
    b = s[s.day <= end]
    sec = b.day.values * 86400.0
    f_out = np.trapezoid(b.f_out_s2, sec)
    phys = np.trapezoid(b.chi_phys, sec)
    num = -(b.s2.iloc[-1] - b.s2.iloc[0]) - f_out - phys
    row.update({'budget_days': float(end), 'chi_phys': phys, 'chi_num': num, 's2_out': f_out,
                'num_share': num / (num + phys) if num + phys > 0 else np.nan,
                'face_flow_over_q': float(-s.face_flow[s.day >= ADJUST].mean() / q) if q > 0 else np.nan})
    fr = r['fr_profile']
    row['Fr_max'] = float(np.nanmax(fr)) if np.isfinite(fr).any() else np.nan
    row['Fr_max_at_m'] = float(r['col_dist'][np.nanargmax(fr)]) if np.isfinite(fr).any() else np.nan
    row['Fr_median'] = float(np.nanmedian(fr)) if np.isfinite(fr).any() else np.nan
    nh, nl, ka = r['halo']
    row['halocline_Ri_lt_025_share'] = nl / nh if nh else np.nan
    row['halocline_AKs_mean'] = ka / nh if nh else np.nan
    a = s[s.day >= ADJUST]
    for c in ('A4_bottom_age_h', 'A3_bottom_age_h', 'pool_age_h', 'pool_river_fraction'):
        if c in s:
            row[f'{c}_mean'] = float(a[c].mean())
            row[f'{c}_end'] = float(s[c].iloc[-1])
    e = s[(s.day >= ADJUST) & (s.pool_end_m > 0)]
    row['pool_end_retreat_m_day'] = -np.polyfit(e.day, e.pool_end_m, 1)[0] if len(e) > 6 else np.nan
    row['salt_left'] = float(s.salt_content.iloc[-1] / s.salt_content.iloc[0])
    row['nan'] = int(s.nan.sum())
    return row


def runs():
    return [(d, scale, f'{prefix}{suffix}') for d, prefix in DOMAINS.items() for scale, suffix in SCALES.items()]


def work(item):
    domain, scale, name = item
    path = EXP / name
    if not finished(path):
        return None
    q = scale * Q_REF / 86400.0
    r = cached(path, q)
    return {'run': name, 'domain': domain, 'scale': scale, 'Q_m3_day': scale * Q_REF, **summarise(r, q)}, r


def scaling(t):
    out = []
    for domain in DOMAINS:
        d = t[(t.domain == domain) & (t.scale > 0)]
        row = {'domain': domain}
        for col in ('pool_rate', 'A4_exchange', 'A3_exchange'):
            v = d[np.isfinite(d[col]) & (d[col] > 0)]
            if len(v) >= 3:
                f = stats.linregress(np.log(v.Q_m3_day), np.log(v[col]))
                half = stats.t.ppf(0.975, len(v) - 2) * f.stderr
                row.update({f'n_{col}': f.slope, f'n_{col}_lo': f.slope - half, f'n_{col}_hi': f.slope + half,
                            f'n_{col}_points': len(v)})
        ok = d[np.isfinite(d.T90)].sort_values('scale')
        fast = ok[ok.T90 <= TC_DAYS]
        if len(fast) and len(ok[ok.T90 > TC_DAYS]):
            hi = fast.iloc[0]
            lo = ok[ok.scale < hi.scale].iloc[-1] if len(ok[ok.scale < hi.scale]) else None
            if lo is not None and lo.T90 > TC_DAYS:
                x = np.interp(np.log(TC_DAYS), np.log([hi.T90, lo.T90]), np.log([hi.Q_m3_day, lo.Q_m3_day]))
                row['Q_c_m3_day'] = float(np.exp(x))
        elif len(fast):
            row['Q_c_m3_day_below'] = float(fast.iloc[0].Q_m3_day)
        out.append(row)
    return pd.DataFrame(out)


def letter(ax, k, text=''):
    ax.text(0.02, 0.96, f'({"abcdefghijkl"[k]}) {text}', transform=ax.transAxes, va='top', fontweight='bold')


def flow_colours():
    scales = [s for s in SCALES if s > 0]
    return dict(zip(scales, plt.cm.viridis(np.linspace(0, 0.9, len(scales)))))


def against_q(ax, t, col, ylabel, floor=False, log=True, ls='-', label=True):
    for domain in DOMAINS:
        d = t[(t.domain == domain) & (t.scale > 0)].sort_values('scale')
        ax.plot(d.Q_m3_day, d[col], marker='o', ms=4, ls=ls, color=DC[domain],
                label=NAMES[domain] if label else None)
        f = t[(t.domain == domain) & (t.scale == 0)]
        if floor and len(f) and np.isfinite(f[col].iloc[0]):
            ax.axhline(f[col].iloc[0], color=DC[domain], ls=':', lw=1)
    ax.set_xscale('log')
    if log:
        ax.set_yscale('log')
    ax.set_xlabel('river flow Q (m³/day)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which='both')


def figures(t, res):
    fc = flow_colours()
    # 1. pool salt against time and against flushing volumes
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    for c, (domain, prefix) in enumerate(DOMAINS.items()):
        for scale, suffix in SCALES.items():
            r = res.get(f'{prefix}{suffix}')
            if r is None:
                continue
            s = r['series']
            p = s.pool_salt / s.pool_salt.iloc[0]
            if scale == 0:
                axes[0, c].plot(s.day, p, color='0.4', ls=':', label='no river')
                continue
            axes[0, c].plot(s.day, p, color=fc[scale], label=f'×{scale:g}')
            axes[1, c].plot(s.tstar, p, color=fc[scale], label=f'×{scale:g}')
        axes[0, c].set_xlabel('day')
        axes[1, c].set_xlabel('flushing volumes t* = Q t / V')
        axes[1, c].set_xscale('log')
        axes[1, c].set_xlim(1e-2, 50)
        letter(axes[0, c], c, NAMES[domain])
        letter(axes[1, c], 3 + c)
    for a in axes.flat:
        a.set_yscale('log')
        a.set_ylim(1e-3, 1.2)
        a.grid(alpha=0.3, which='both')
    for a in axes[:, 0]:
        a.set_ylabel(f'pool salt (deeper than {POOL_DEPTH:g} m) / initial')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'flow_pool.png', dpi=130)
    plt.close(fig)

    # 2. rates, flushing time and age against Q
    tgt, lo, hi = cm.target()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    against_q(axes[0, 0], t, 'pool_rate', 'pool loss rate (%/day)', floor=True)
    against_q(axes[0, 1], t, 'A4_exchange', 'exchange rate (%/day): A4 solid, A3 dashed', floor=True)
    against_q(axes[0, 1], t, 'A3_exchange', 'exchange rate (%/day): A4 solid, A3 dashed', ls='--', label=False)
    axes[0, 1].axhspan(lo, hi, color=OBS_C, alpha=0.6, lw=0, label='observed (A4)')
    against_q(axes[1, 0], t, 'T50', 'T50 of the pool salt (days)')
    against_q(axes[1, 1], t, 'A4_bottom_age_h_mean', 'bottom ventilation age (h): A4 solid, A3 dashed')
    against_q(axes[1, 1], t, 'A3_bottom_age_h_mean', 'bottom ventilation age (h): A4 solid, A3 dashed', ls='--',
              label=False)
    for k, a in enumerate(axes.flat):
        letter(a, k)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'flow_rates.png', dpi=130)
    plt.close(fig)

    # 3. mixing: numerical share, the budget terms, regime numbers
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    against_q(axes[0, 0], t, 'num_share', 'numerical share of the mixing', log=False)
    against_q(axes[0, 1], t, 'chi_phys', 'mixing until the pool is gone (PSU² m³): physical solid, numerical dashed')
    against_q(axes[0, 1], t, 'chi_num', 'mixing until the pool is gone (PSU² m³): physical solid, numerical dashed',
              ls='--', label=False)
    against_q(axes[1, 0], t, 'Fr_max', 'densimetric Froude number: maximum solid, median dashed')
    against_q(axes[1, 0], t, 'Fr_median', 'densimetric Froude number: maximum solid, median dashed', ls='--',
              label=False)
    axes[1, 0].axhline(1, color='0.4', lw=0.8)
    against_q(axes[1, 1], t, 'halocline_Ri_lt_025_share', 'halocline cells with Ri < 0.25 (share)', log=False)
    for k, a in enumerate(axes.flat):
        letter(a, k)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'flow_mixing.png', dpi=130)
    plt.close(fig)

    # 4. shape: ratios to NoMeander
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    base = t[t.domain == 'NoMeander'].set_index('scale')
    for a, (col, label) in zip(axes.flat, [('pool_rate', 'pool loss rate'), ('T50', 'T50'),
                                          ('A4_exchange', 'A4 exchange rate'), ('num_share', 'numerical share')]):
        for domain in ('Meander', 'Alexander'):
            d = t[(t.domain == domain) & (t.scale > 0)].set_index('scale')
            ratio = (d[col] / base[col].reindex(d.index)).sort_index()
            a.plot(ratio.index * Q_REF, ratio.values, marker='o', ms=4, color=DC[domain],
                   label=f'{NAMES[domain]} / No-Meander')
        a.axhline(1, color='0.4', lw=0.8)
        if len(base):
            a.set_xscale('log')
        a.set_xlabel('river flow Q (m³/day)')
        a.set_ylabel(f'{label}, ratio')
        a.grid(alpha=0.3, which='both')
    for k, a in enumerate(axes.flat):
        letter(a, k)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'flow_shape.png', dpi=130)
    plt.close(fig)

    # 5. where the salt stays: upstream end of the pool, salt left per reach at t* = 1
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey='row')
    for c, (domain, prefix) in enumerate(DOMAINS.items()):
        for scale, suffix in SCALES.items():
            r = res.get(f'{prefix}{suffix}')
            if r is None or scale == 0:
                continue
            s = r['series']
            axes[0, c].plot(s.tstar, s.pool_end_m / 1e3, color=fc[scale], label=f'×{scale:g}')
            if 1 in r['reach_left']:
                axes[1, c].plot(r['reach_x'] / 1e3, r['reach_left'][1], color=fc[scale], marker='o', ms=3,
                                label=f'×{scale:g}')
        axes[0, c].set_xscale('log')
        axes[0, c].set_xlim(1e-2, 50)
        axes[0, c].set_xlabel('flushing volumes t*')
        axes[1, c].set_xlabel('distance from the mouth (km)')
        letter(axes[0, c], c, NAMES[domain])
        letter(axes[1, c], 3 + c)
    axes[0, 0].set_ylabel(f'upstream end of the pool, bed S ≥ {S_POOL_END:g} (km)')
    axes[1, 0].set_ylabel(f'salt left per {REACH_M:g} m reach at t* = 1')
    axes[0, 0].legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'flow_where.png', dpi=130)
    plt.close(fig)

    # 6. centreline sections at equal t*, one figure per flow
    for suffix in SECTION_RUNS:
        fig, axes = plt.subplots(len(SECTION_TSTAR), 3, figsize=(17, 2.6 * len(SECTION_TSTAR)), sharex=True,
                                 sharey=True)
        mesh = None
        for c, (domain, prefix) in enumerate(DOMAINS.items()):
            r = res.get(f'{prefix}{suffix}')
            for k, ts in enumerate(SECTION_TSTAR):
                a = axes[k, c]
                if r is None or ts not in r['sections']:
                    a.set_axis_off()
                    continue
                s, z = r['sections'][ts]
                x = np.broadcast_to(r['col_dist'] / 1e3, z.shape)
                mesh = a.pcolormesh(x, z, s, vmin=0, vmax=20, cmap='viridis', shading='nearest')
                letter(a, 3 * k + c, f'{NAMES[domain]}, t* = {ts:g}')
        for a in axes[-1]:
            a.set_xlabel('distance from the mouth (km)')
        for a in axes[:, 0]:
            a.set_ylabel('z (m)')
        if mesh is not None:
            fig.colorbar(mesh, ax=axes, shrink=0.8, label='salinity')
        fig.savefig(OUT / f'flow_sections_{suffix}.png', dpi=120)
        plt.close(fig)


def manuscript_figure(t, res):
    '''Fig. 5: the river-flow experiment on the three domains, in the style of Fig. 4 (compare_wind_iw.py).
    Station rates only where the daily method holds (as in Fig. 4); positions by distance from the mouth.'''
    from matplotlib.lines import Line2D
    from analyse_runs import REPO, STATIONS
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.32, wspace=0.42)
    top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    t = t[t.scale > 0].copy()
    for st in ('A4', 'A3'):
        t.loc[t[f'{st}_method'] != 'daily', f'{st}_exchange'] = np.nan
    t['num_pct'] = 100 * t.num_share
    tgt, lo, hi = cm.target()
    dist = {st: f'{d:.0f} m' for st, d in STATIONS.items()}
    panels = [('A4_exchange', f'exchange rate, {dist["A4"]} (% day$^{{-1}}$)', False),
              ('A3_exchange', f'exchange rate, {dist["A3"]} (% day$^{{-1}}$)', False),
              ('pool_rate', 'salt loss rate, bottom pool\n(% day$^{-1}$)', True),
              ('num_pct', 'numerical share of the mixing (%)', False)]
    for k, (a, (col, label, log)) in enumerate(zip(top, panels)):
        for domain in DOMAINS:
            d = t[t.domain == domain].sort_values('scale')
            a.plot(d.Q_m3_day / 1e3, d[col], marker='o', ms=5, lw=1.8, color=DC[domain])
        if col == 'A4_exchange':
            a.axhspan(lo, hi, color=OBS_C, alpha=0.7, lw=0)
        a.axvline(Q_REF / 1e3, color='#d62728', lw=2.5, zorder=0)
        a.set_xscale('log')
        if log:
            a.set_yscale('log')
        if col == 'num_pct':
            a.set_ylim(0, 100)
        a.set_ylabel(label)
        a.set_xlabel('river flow Q (10$^3$ m$^3$ day$^{-1}$)')
        a.set_xticks([10, 100, 1000])
        a.grid(alpha=0.3, which='both')
        y, va = (0.04, 'bottom') if col == 'num_pct' else (0.96, 'top')
        a.text(0.04, y, f'({"abcd"[k]})', transform=a.transAxes, va=va, fontweight='bold', fontsize=14)
    for a in top[:2]:
        a.set_xlim(3, 130)
        a.set_xticks([10, 100])
    fc = flow_colours()
    bottom = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4], sharey=None)]
    for scale, suffix in SCALES.items():
        r = res.get(f'alex_{suffix}')
        if r is None or scale == 0:
            continue
        s = r['series']
        p = s.pool_salt / s.pool_salt.iloc[0]
        bottom[0].plot(s.day, p, color=fc[scale], lw=2, label=f'×{scale:g}')
        bottom[1].plot(s.tstar, p, color=fc[scale], lw=2)
    bottom[0].set_xlabel('time (days)')
    bottom[0].set_xlim(0, 30)
    bottom[1].set_xlabel('flushing volumes, Q t / V')
    bottom[1].set_xlim(0, 8)
    for k, a in enumerate(bottom):
        a.set_yscale('log')
        a.set_ylim(1e-3, 4)
        a.set_ylabel('bottom-pool salt, fraction of initial')
        a.grid(alpha=0.3, which='both')
        a.text(0.015, 0.96, f'({"ef"[k]})', transform=a.transAxes, va='top', fontweight='bold', fontsize=14)
    hq, lq = bottom[0].get_legend_handles_labels()
    fig.legend(hq, lq, title='river flow of the curves in (e, f), Q / Q$_{ref}$ (Alexander domain)', loc='lower center',
               ncol=len(lq), fontsize=13, title_fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.0,
               columnspacing=1.4)
    handles = [Line2D([], [], color=DC[d], lw=3, marker='o', label=NAMES[d]) for d in DOMAINS]
    handles += [Line2D([], [], color='#d62728', lw=2.5, label='reference flow'),
                Line2D([], [], color=OBS_C, lw=7, label='observed')]
    fig.legend(handles=handles, loc='upper center', ncol=5, mode='expand', fontsize=15, frameon=False,
               bbox_to_anchor=(0.1, 0.96, 0.8, 0.04), handlelength=2.6)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    for a in top + bottom:
        for ax_ in (a.xaxis, a.yaxis):
            if ax_.get_scale() == 'log':
                ax_.set_major_formatter(plain)
                ax_.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.subplots_adjust(left=0.07, right=0.98, top=0.915, bottom=0.13)
    out = REPO / 'docs' / 'figures' / 'manuscript_plot5.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'written {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--experiment', default='flow_bar', help='run folder under ROMS_runs')
    a = ap.parse_args()
    set_experiment(a.experiment)
    OUT.mkdir(parents=True, exist_ok=True)
    items = runs()
    with Pool(a.jobs, initializer=set_experiment, initargs=(a.experiment,)) as pool:
        out = pool.map(work, items)
    rows, res = [], {}
    for item, o in zip(items, out):
        if o is None:
            print(f'{item[2]}: not finished, skipped')
            continue
        rows.append(o[0])
        res[item[2]] = o[1]
    if not rows:
        raise SystemExit(f'no finished runs in {EXP}')
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'flow.csv', index=False, float_format='%.4g')
    sc = scaling(t)
    sc.to_csv(OUT / 'flow_scaling.csv', index=False, float_format='%.4g')
    cols = ['run', 'pool_rate', 'T50', 'T90', 'A4_exchange', 'A3_exchange', 'num_share', 'Fr_max',
            'halocline_Ri_lt_025_share', 'A4_bottom_age_h_mean', 'face_flow_over_q', 'salt_left', 'nan']
    with pd.option_context('display.width', 250):
        print(t[[c for c in cols if c in t]].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
        print(sc.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    figures(t, res)
    manuscript_figure(t, res)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
