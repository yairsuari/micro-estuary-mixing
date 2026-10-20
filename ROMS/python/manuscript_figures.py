"""
Figures for the manuscript draft (written to docs/figures/):
    manuscript_plot1.png           (Fig. 1) the three model domains: plan view (centreline coloured by depth, banks,
                                   stations A4 and A3), the along-channel width, and the initial salinity
                                   with the vertical grid along the Alexander centreline (ini.nc, the state
                                   from which all experiments start; the bathymetry is the same in all three
                                   domains). Distances from the mouth are measured from the bar crest
                                   (mouth_index); closed-mouth grids:
                                   ROMS_GRIDS=grids_bar ROMS_DATA=Data_bar python manuscript_figures.py
    manuscript_plot2.png           (Fig. 2) the observations at A4 (and A3): (a) the fresh-water-only period
                                   P21, (b) a small and a large flood, (c) histogram of the daily exchange
                                   rates of all periods (inputs from mixing_periods.py: periods.csv,
                                   daily_exchange.csv)
    figS1_initial_salinity.png     (no longer in the manuscript) initial salinity of the three domains

Usage: python manuscript_figures.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

from common import REPO, DOMAINS, domain_dir, mouth_index
import mixing_periods as mp

OUT = REPO / 'docs' / 'figures'
plt.rcParams.update({'font.size': 11})
LABEL = {'Alexander': 'Alexander', 'Meander': 'Meander', 'NoMeander': 'No-Meander'}
STATIONS = {'A4': 400.0, 'A3': 3500.0}
DEPTH_CMAP = 'viridis_r'
BAR_COLOUR = '#8b5a2b'      # the sandbar / bed colour, as in the section panel
BAR_HALF_WIDTH_KM = 0.16    # Fig. 1e: half width at the base of the sandbar as drawn (exaggerated)


def panel_label(ax, text, loc='upper left'):
    '''panel letter (and short name) inside the axes, instead of a title above it.'''
    x, ha = (0.015, 'left') if 'left' in loc else (0.985, 'right')
    y, va = (0.97, 'top') if 'upper' in loc else (0.03, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=12, fontweight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5), zorder=10)


def centreline(g):
    '''centre row, its x and y, and the distance along it from the mouth (the bar crest, negative at sea)'''
    j = g.sizes['eta_rho'] // 2
    x, y = g.x_rho.values[j], g.y_rho.values[j]
    dist = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return j, x, y, dist - dist[mouth_index(g)]


def domains_figure():
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], height_ratios=[2.2, 1.1, 1.0], hspace=0.22, wspace=0.16)
    norm = plt.Normalize(0, 3.5)
    prof = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1:, 1])]
    colours = {'Alexander': '#1f5fa8', 'Meander': '#e08a1e', 'NoMeander': '#2a9d4b'}
    for k, name in enumerate(DOMAINS):
        g = xr.open_dataset(domain_dir(name) / 'grid.nc')
        j, x, y, dist = centreline(g)
        m = mouth_index(g)
        x0, y0 = x[m], y[m]
        ax = fig.add_subplot(gs[k, 0])
        for jb in (1, g.sizes['eta_rho'] - 2):                       # outermost computed rows
            ax.plot((g.x_rho.values[jb, m:] - x0) / 1000, (g.y_rho.values[jb, m:] - y0) / 1000, color='0.35',
                    lw=0.6)
        pts = np.column_stack([(x[m:] - x0) / 1000, (y[m:] - y0) / 1000]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=DEPTH_CMAP, norm=norm, lw=3.5)
        lc.set_array(0.5 * (g.h.values[j, m + 1:] + g.h.values[j, m:-1]))
        ax.add_collection(lc)
        # the sandbar at the mouth (the crest row), and the river inflow at the head
        if name != 'Alexander':
            ax.plot(0, 0, 's', mfc=BAR_COLOUR, mec='k', ms=8, zorder=5)
            ax.annotate('sandbar', (0, 0), xytext=(-9, -2), textcoords='offset points', fontsize=10, ha='right',
                        va='top')
        xh, yh = (x[-1] - x0) / 1000, (y[-1] - y0) / 1000
        ax.plot(xh, yh, 'v', color='k', ms=7, zorder=5)
        ax.annotate('river', (xh, yh), xytext=(4, 4), textcoords='offset points', fontsize=10)
        if name == 'Alexander':
            for st, d in STATIONS.items():
                i = m + int(np.argmin(np.abs(dist[m:] - d)))
                ax.plot((x[i] - x0) / 1000, (y[i] - y0) / 1000, 'o', mfc='white', mec='k', ms=6)
                ax.annotate(st, ((x[i] - x0) / 1000, (y[i] - y0) / 1000), xytext=(5, 5),
                            textcoords='offset points', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()
        ax.margins(0.06)
        if name == 'NoMeander':
            ax.set_ylim(-0.3, 0.3)
        panel_label(ax, f'({"abc"[k]}) {LABEL[name]}', loc='upper right' if name == 'Alexander' else 'upper left')
        if k == 2:
            ax.set_xlabel('east (km)')
        ax.set_ylabel('north (km)')
        ax.grid(alpha=0.3)
        # along-channel width (outermost computed rows, ~ the wet width) and centreline depth
        wb = np.hypot(g.x_rho.values[1] - g.x_rho.values[-2], g.y_rho.values[1] - g.y_rho.values[-2])
        prof[0].plot(dist[m:] / 1000, wb[m:], color=colours[name], lw=1.3, label=LABEL[name])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DEPTH_CMAP)
    cax = fig.add_axes([0.13, 0.04, 0.3, 0.013])
    fig.colorbar(sm, cax=cax, orientation='horizontal').set_label('centreline depth (m)')
    prof[0].set_ylabel('wet width (m)')
    panel_label(prof[0], '(d)', loc='upper left')
    section_panel(fig, prof[1])
    for a in prof[:1]:
        a.grid(alpha=0.3)
        a.legend(fontsize=10, loc='upper right')
    fig.savefig(OUT / 'manuscript_plot1.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def section(name):
    '''centreline distance (km), level interfaces z_w and centres z_r, salinity and depth of ini.nc.'''
    g = xr.open_dataset(domain_dir(name) / 'grid.nc')
    d = xr.open_dataset(domain_dir(name) / 'ini.nc', decode_times=False)
    j, x, y, dist = centreline(g)
    h = g.h.values[j]
    zeta = d.zeta.values[0, j]
    hc, n = float(d.hc), d.sizes['s_rho']
    def z(s, C):
        z0 = (hc * s[:, None] + h[None, :] * C[:, None]) / (hc + h[None, :])
        return zeta[None, :] + (zeta[None, :] + h[None, :]) * z0
    zr = z((np.arange(1, n + 1) - n - 0.5) / n, d.Cs_r.values)
    zw = z(np.arange(0, n + 1) / n - 1.0, d.Cs_w.values)
    return dist / 1000, zr, zw, d.salt.values[0, :, j, :], h, zeta


def draw_section(ax, km, zr, zw, sal, h, zeta, levels, xlim=None):
    z = np.vstack([-h[None, :], zr, zeta[None, :]])
    S = np.vstack([sal[:1], sal, sal[-1:]])
    X = np.broadcast_to(km, z.shape)
    cf = ax.contourf(X, z, S, levels=levels, cmap='Blues', extend='max')
    for k in range(zw.shape[0]):
        ax.plot(km, zw[k], color='k', lw=0.35)
    ax.fill_between(km, -h, -4.5, color=BAR_COLOUR, zorder=3)
    ax.plot(km, -h, color='k', lw=2.2, zorder=4, solid_joinstyle='miter')
    if xlim:
        ax.set_xlim(*xlim)
    return cf


def mouth_cells(ax, km, zw, sal, h, levels, n_cells=7):
    '''the first cells from the mouth drawn cell by cell: salinity, grid and the stepped bottom.'''
    m = km * 1000
    edges = np.concatenate(([m[0] - (m[1] - m[0]) / 2], (m[1:] + m[:-1]) / 2, [m[-1] + (m[-1] - m[-2]) / 2]))
    norm = matplotlib.colors.BoundaryNorm(levels, plt.get_cmap('Blues').N, extend='max')
    for i in range(n_cells):
        for k in range(sal.shape[0]):
            ax.fill_between([edges[i], edges[i + 1]], zw[k, i], zw[k + 1, i], color=plt.get_cmap('Blues')(norm(sal[k, i])),
                            ec='k', lw=0.35)
        ax.fill_between([edges[i], edges[i + 1]], -h[i], -4.5, color='#8b5a2b', ec='k', lw=0.8)
    ax.set_xlim(edges[0], edges[n_cells])
    ax.annotate(f'sandbar (depth {h[0]:.2f} m, dry)', ((edges[0] + edges[1]) / 2, -h[0]), xytext=(0.25, 0.78),
                textcoords='axes fraction', fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.7))


def section_panel(fig, ax):
    km, zr, zw, sal, h, zeta = section('Alexander')
    levels = np.arange(0, 32, 2)
    cf = draw_section(ax, km, zr, zw, sal, h, zeta, levels)
    m = int(np.argmin(np.abs(km)))
    # the sandbar is one cell wide (invisible at this scale): drawn as a wider brown mound for visibility
    #; the caption says so
    half = BAR_HALF_WIDTH_KM
    ax.fill([-half, -half / 4, half / 4, half], [-3.8, -h[m], -h[m], -3.8], color=BAR_COLOUR, ec='k', lw=1.2,
            zorder=6)
    ax.annotate('sandbar', (half / 2, -h[m] - 0.3), xytext=(0.35, -0.9), fontsize=10,
                bbox=dict(fc='white', ec='none', pad=1), arrowprops=dict(arrowstyle='->', lw=0.8), zorder=7)
    for st, d in STATIONS.items():                                  # the salinity stations
        ax.axvline(d / 1000, color='k', ls='--', lw=1.1, zorder=5)
        ax.text(d / 1000, 0.27, f' {st}', ha='left', va='top', fontsize=10, zorder=6,
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    ax.set_xlim(-BAR_HALF_WIDTH_KM, km[-1])
    ax.set_ylim(-3.8, 0.3)
    ax.set_xlabel('distance from the mouth (km)')
    ax.set_ylabel('depth (m)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{abs(v):g}'))   # depth, no minus
    ax.yaxis.set_label_coords(-0.085, 0.5)
    panel_label(ax, '(e)', loc='lower right')
    cb = fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
    cb.set_label('salinity (PSU)')


def initial_salinity_figure():
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    levels = np.arange(0, 32, 2)
    for k, (name, ax) in enumerate(zip(DOMAINS, axes)):
        g = xr.open_dataset(domain_dir(name) / 'grid.nc')
        d = xr.open_dataset(domain_dir(name) / 'ini.nc', decode_times=False)
        j, x, y, dist = centreline(g)
        h = g.h.values[j]
        zeta = d.zeta.values[0, j]
        hc = float(d.hc)
        n = d.sizes['s_rho']
        s = (np.arange(1, n + 1) - n - 0.5) / n
        z0 = (hc * s[:, None] + h[None, :] * d.Cs_r.values[:, None]) / (hc + h[None, :])
        z = zeta[None, :] + (zeta[None, :] + h[None, :]) * z0
        sal = d.salt.values[0, :, j, :]
        z = np.vstack([-h[None, :], z, zeta[None, :]])                 # extend to the bed and the surface
        sal = np.vstack([sal[:1], sal, sal[-1:]])
        X = np.broadcast_to(dist / 1000, z.shape)
        cf = ax.contourf(X, z, sal, levels=levels, cmap='Blues', extend='max')
        ax.fill_between(dist / 1000, -h, -4.2, color='#8b5a2b')
        ax.set_ylim(-4, 0.2)
        ax.set_title(f'({"abc"[k]}) {LABEL[name]}', loc='left', fontsize=10)
        ax.set_ylabel('depth (m)')
    axes[-1].set_xlabel('distance from the mouth along the centreline (km)')
    fig.colorbar(cf, ax=axes, shrink=0.8).set_label('salinity')
    fig.savefig(OUT / 'figS1_initial_salinity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


STEADY_PERIOD = 'P21'                                    # longest fresh-water-only period, R^2 0.99 at A4 and A3
FLOODS = {'small flood': '2023-12-15', 'large flood': '2019-12-27'}
SURF, BOT = '#1f5fa8', '#b2392f'


def flood_onset(s, day):
    '''first time the despiked A4 surface salinity falls below 2 within a day of the event date.'''
    t = pd.Timestamp(day)
    w = s.surface[t - pd.Timedelta('1D'):t + pd.Timedelta('1D')]
    return w[w < 2].index[0]


def observations_figure():
    '''(a) a fresh-water-only period, (b) a small and a large flood, (c) the daily exchange rates at A4.'''
    a4 = mp.despike(mp.load_station('A4'), 5)
    a3 = mp.despike(mp.load_station('A3'), 5)
    periods = pd.read_csv(mp.INPUT / 'periods.csv', parse_dates=['start', 'end']).set_index('id')
    daily_ex = pd.read_csv(mp.INPUT / 'daily_exchange.csv')

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={'hspace': 0.28})
    ax = axes[0]
    p = periods.loc[STEADY_PERIOD]
    t0, t1 = p.start - pd.Timedelta('3D'), p.end + pd.Timedelta('4D')
    for d, name, ls in ((a4, 'A4', '-'), (a3, 'A3', ':')):
        w = d[t0:t1]
        ax.plot(w.index, w.surface, color=SURF, lw=0.8, ls=ls, label=f'{name} surface')
        ax.plot(w.index, w.bottom, color=BOT, lw=0.8, ls=ls, label=f'{name} bottom')
    ax.axvspan(p.start, p.end + pd.Timedelta('1D'), color='0.9', zorder=0)
    ax.set_ylabel('salinity (PSU)')
    panel_label(ax, '(a)', loc='lower left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(fontsize=10, ncol=4, loc='center right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    for (label, day), ls in zip(FLOODS.items(), ('--', '-')):
        t = flood_onset(a4, day)
        w = a4[t - pd.Timedelta('2D'):t + pd.Timedelta('8D')]
        h = (w.index - t) / pd.Timedelta('1D')
        ax.plot(h, w.surface, color=SURF, lw=1, ls=ls, label=f'{label} ({t:%d %b %Y}), surface')
        ax.plot(h, w.bottom, color=BOT, lw=1, ls=ls, label=f'{label}, bottom')
    ax.axvline(0, color='0.4', lw=0.7)
    ax.set_xlabel('days from the flood onset at A4')
    ax.set_ylabel('salinity (PSU)')
    panel_label(ax, '(b)', loc='upper right')
    ax.legend(fontsize=9, loc='center right')
    ax.grid(alpha=0.3)

    ax = axes[2]
    v = daily_ex.exchange.dropna()
    ax.hist(v, bins=np.arange(np.floor(v.min() / 2) * 2, np.ceil(v.max() / 2) * 2 + 2, 2), color='#9cb8d9',
            edgecolor='white')
    ax.axvline(v.mean(), color='#1f5fa8', lw=2, label=f'mean {v.mean():.1f} % day⁻¹')
    ax.axvline(v.median(), color='#1f5fa8', lw=1.5, ls='--', label=f'median {v.median():.1f} % day⁻¹')
    ax.axvline(0, color='0.3', lw=0.8)
    ax.set_xlabel('daily exchange rate at A4 (% day$^{-1}$)')
    ax.set_ylabel('days')
    lo, hi = -5, 17.5
    ax.set_xlim(lo, hi)
    out = int(((v < lo) | (v > hi)).sum())
    print(f'Fig. 2c: {out} of {len(v)} daily rates outside {lo}..{hi} %/day '
          f'({int((v < lo).sum())} below, {int((v > hi).sum())} above)')
    panel_label(ax, '(c)', loc='upper left')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    fig.savefig(OUT / 'manuscript_plot2.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    domains_figure()
    observations_figure()
    print(OUT)
