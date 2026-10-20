"""
Write ROMS grid files (Data/<domain>/grid.nc) from the pygridgen grids in
GRIDATA/grids/<domain>_grid.npz.

The pygridgen points become the ROMS rho-points, one to one with the POM
T-cells:
    xi  (i) along the channel, xi_rho = IM, i=0 at the mouth (western open boundary)
    eta (j) across the channel, eta_rho = JM
The outer rho rows/columns are ROMS boundary points. The north/south walls lie
on the v-faces between rows 0|1 and JM-2|JM-1, so the computed (interior)
channel is rows 1..JM-2 (5 cells for JM=7), the same as the wet POM columns.

Masks are 1 everywhere: the side walls are the closed boundaries and drying
is handled by WET_DRY.

Usage: python roms_grid.py [--domains Alexander Meander NoMeander]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (DOMAINS, GRIDATA, EARTH_RADIUS, OMEGA, domain_dir, estuary_slice, load_grid,
                    rho_to_u, rho_to_v, rho_to_psi)

sys.path.insert(0, str(GRIDATA))
from grid_quality import quality        # noqa: E402  (shared with the POM grids)


def spacing(x, y, axis):
    '''Grid spacing at rho points along an axis: centred differences, one-sided at the edges.'''
    d = np.zeros_like(x)
    if axis == 1:
        d[:, 1:-1] = 0.5 * np.hypot(x[:, 2:] - x[:, :-2], y[:, 2:] - y[:, :-2])
        d[:, 0] = np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])
        d[:, -1] = np.hypot(x[:, -1] - x[:, -2], y[:, -1] - y[:, -2])
    else:
        d[1:-1, :] = 0.5 * np.hypot(x[2:, :] - x[:-2, :], y[2:, :] - y[:-2, :])
        d[0, :] = np.hypot(x[1, :] - x[0, :], y[1, :] - y[0, :])
        d[-1, :] = np.hypot(x[-1, :] - x[-2, :], y[-1, :] - y[-2, :])
    return d


def grid_angle(lon, lat):
    '''Angle (rad) between the xi axis and true east, from lon/lat (centred differences).'''
    lonr, latr = np.radians(lon), np.radians(lat)
    dlon = np.zeros_like(lon)
    dlat = np.zeros_like(lat)
    dlon[:, 1:-1] = lonr[:, 2:] - lonr[:, :-2]
    dlat[:, 1:-1] = latr[:, 2:] - latr[:, :-2]
    dlon[:, 0], dlat[:, 0] = lonr[:, 1] - lonr[:, 0], latr[:, 1] - latr[:, 0]
    dlon[:, -1], dlat[:, -1] = lonr[:, -1] - lonr[:, -2], latr[:, -1] - latr[:, -2]
    return np.arctan2(dlat, dlon * np.cos(latr))


def build_grid(domain):
    g = load_grid(domain)
    x, y, lon, lat, h = g['x'], g['y'], g['lon'], g['lat'], g['h']
    eta_rho, xi_rho = h.shape

    dx = spacing(x, y, axis=1)          # along xi
    dy = spacing(x, y, axis=0)          # along eta
    pm, pn = 1.0 / dx, 1.0 / dy
    dndx = np.zeros_like(h)
    dmde = np.zeros_like(h)
    dndx[:, 1:-1] = 0.5 * (1.0 / pn[:, 2:] - 1.0 / pn[:, :-2])
    dmde[1:-1, :] = 0.5 * (1.0 / pm[2:, :] - 1.0 / pm[:-2, :])
    angle = grid_angle(lon, lat)
    f = 2.0 * OMEGA * np.sin(np.radians(lat))

    # right-handed check: eta must point 90 deg counter-clockwise from xi
    xi_vec = np.stack([x[:-1, 1:] - x[:-1, :-1], y[:-1, 1:] - y[:-1, :-1]])
    eta_vec = np.stack([x[1:, :-1] - x[:-1, :-1], y[1:, :-1] - y[:-1, :-1]])
    cross = xi_vec[0] * eta_vec[1] - xi_vec[1] * eta_vec[0]
    assert np.all(cross > 0), f'{domain}: grid is not right-handed everywhere'
    for name, a in (('pm', pm), ('pn', pn), ('h', h), ('angle', angle)):
        assert np.all(np.isfinite(a)), f'{domain}: {name} has non-finite values'
    assert np.all(pm > 0) and np.all(pn > 0)

    rho = ('eta_rho', 'xi_rho')
    u = ('eta_u', 'xi_u')
    v = ('eta_v', 'xi_v')
    psi = ('eta_psi', 'xi_psi')

    def var(dims, data, long_name, units=None, field=None):
        attrs = {'long_name': long_name}
        if units:
            attrs['units'] = units
        if field:
            attrs['field'] = field
        return (dims, np.asarray(data, dtype='f8'), attrs)

    ds = xr.Dataset(
        {
            'spherical': ((), np.int32(1), {'long_name': 'grid type logical switch',
                                            'flag_values': '0, 1',
                                            'flag_meanings': 'Cartesian spherical'}),
            'xl': ((), np.float64(np.median(dx.sum(axis=1))),
                   {'long_name': 'domain length in the XI-direction', 'units': 'meter'}),
            'el': ((), np.float64(np.median(dy.sum(axis=0))),
                   {'long_name': 'domain length in the ETA-direction', 'units': 'meter'}),
            'h': var(rho, h, 'bathymetry at RHO-points', 'meter', 'bath, scalar'),
            'f': var(rho, f, 'Coriolis parameter at RHO-points', 'second-1', 'coriolis, scalar'),
            'pm': var(rho, pm, 'curvilinear coordinate metric in XI', 'meter-1', 'pm, scalar'),
            'pn': var(rho, pn, 'curvilinear coordinate metric in ETA', 'meter-1', 'pn, scalar'),
            'dndx': var(rho, dndx, 'xi derivative of inverse metric factor pn', 'meter', 'dndx, scalar'),
            'dmde': var(rho, dmde, 'eta derivative of inverse metric factor pm', 'meter', 'dmde, scalar'),
            'angle': var(rho, angle, 'angle between XI-axis and EAST', 'radians', 'angle, scalar'),
            'x_rho': var(rho, x, 'x-locations of RHO-points (ITM, EPSG:6984)', 'meter'),
            'y_rho': var(rho, y, 'y-locations of RHO-points (ITM, EPSG:6984)', 'meter'),
            'x_u': var(u, rho_to_u(x), 'x-locations of U-points (ITM)', 'meter'),
            'y_u': var(u, rho_to_u(y), 'y-locations of U-points (ITM)', 'meter'),
            'x_v': var(v, rho_to_v(x), 'x-locations of V-points (ITM)', 'meter'),
            'y_v': var(v, rho_to_v(y), 'y-locations of V-points (ITM)', 'meter'),
            'x_psi': var(psi, rho_to_psi(x), 'x-locations of PSI-points (ITM)', 'meter'),
            'y_psi': var(psi, rho_to_psi(y), 'y-locations of PSI-points (ITM)', 'meter'),
            'lon_rho': var(rho, lon, 'longitude of RHO-points', 'degree_east', 'lon_rho, scalar'),
            'lat_rho': var(rho, lat, 'latitude of RHO-points', 'degree_north', 'lat_rho, scalar'),
            'lon_u': var(u, rho_to_u(lon), 'longitude of U-points', 'degree_east', 'lon_u, scalar'),
            'lat_u': var(u, rho_to_u(lat), 'latitude of U-points', 'degree_north', 'lat_u, scalar'),
            'lon_v': var(v, rho_to_v(lon), 'longitude of V-points', 'degree_east', 'lon_v, scalar'),
            'lat_v': var(v, rho_to_v(lat), 'latitude of V-points', 'degree_north', 'lat_v, scalar'),
            'lon_psi': var(psi, rho_to_psi(lon), 'longitude of PSI-points', 'degree_east', 'lon_psi, scalar'),
            'lat_psi': var(psi, rho_to_psi(lat), 'latitude of PSI-points', 'degree_north', 'lat_psi, scalar'),
            'mask_rho': var(rho, np.ones_like(h), 'mask on RHO-points', None, 'mask_rho, scalar'),
            'mask_u': var(u, np.ones((eta_rho, xi_rho - 1)), 'mask on U-points', None, 'mask_u, scalar'),
            'mask_v': var(v, np.ones((eta_rho - 1, xi_rho)), 'mask on V-points', None, 'mask_v, scalar'),
            'mask_psi': var(psi, np.ones((eta_rho - 1, xi_rho - 1)), 'mask on PSI-points', None, 'mask_psi, scalar'),
        },
        attrs={
            'type': 'ROMS GRID file',
            'title': f'{domain} micro-estuary grid ({xi_rho}x{eta_rho} rho points)',
            'source': 'GRIDATA/make_domains.py + GRIDATA/grid_IC.py (pygridgen), ROMS/python/roms_grid.py',
            'notes': 'xi along the channel from the western open boundary (xi=0); the mouth (bar or '
                     'crest) is at xi=mouth_index, the estuary beyond it; eta across the channel; '
                     'outer rho rows are the closed side walls.',
            'mouth_index': np.int32(g['mouth_index']),
            'history': f'created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC',
        },
    )
    for mask in ('mask_rho', 'mask_u', 'mask_v', 'mask_psi'):
        ds[mask].attrs['flag_values'] = '0., 1.'
        ds[mask].attrs['flag_meanings'] = 'land water'

    # estuary cells: rows 1..eta-2, from the cell after the mouth to xi-2
    inner = estuary_slice(g)
    area_interior = np.sum(1.0 / (pm[inner] * pn[inner]))
    vol_interior = np.sum(h[inner] / (pm[inner] * pn[inner]))
    q = quality(x.T, y.T, h.T)
    summary = {
        'domain': domain,
        'xi_rho': int(xi_rho), 'eta_rho': int(eta_rho), 'mouth_index': int(g['mouth_index']),
        'dx_m': [float(dx.min()), float(np.median(dx)), float(dx.max())],
        'dy_m': [float(dy.min()), float(np.median(dy)), float(dy.max())],
        'h_m': [float(h.min()), float(h.max())],
        'angle_min_deg': float(q['angle'].min()),
        'barotropic_dt_min_s': float(q['dt'].min()),
        'interior_area_m2': float(area_interior),
        'interior_volume_m3': float(vol_interior),
        'interior_mean_depth_m': float(vol_interior / area_interior),
    }
    return ds, summary


def quicklook(ds, domain, out):
    fields = [('h', 'Depth h (m)', 'Blues'),
              ('dx', 'dx = 1/pm (m)', 'viridis'),
              ('dy', 'dy = 1/pn (m)', 'viridis'),
              ('angle_deg', 'angle (deg from east)', 'twilight')]
    data = {'h': ds.h.values, 'dx': 1 / ds.pm.values, 'dy': 1 / ds.pn.values,
            'angle_deg': np.degrees(ds.angle.values)}
    fig, axes = plt.subplots(len(fields), 1, figsize=(11, 7), sharex=True)
    for ax, (key, title, cmap) in zip(axes, fields):
        pc = ax.pcolormesh(data[key], cmap=cmap, shading='nearest')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_ylabel('eta', fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(pc, ax=ax, pad=0.01)
    axes[-1].set_xlabel('xi (0 = mouth)', fontsize=8)
    fig.suptitle(f'{domain}: ROMS grid in index space', x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(domains=DOMAINS):
    for domain in domains:
        ds, summary = build_grid(domain)
        out_dir = domain_dir(domain)
        encoding = {v: {'_FillValue': None} for v in ds.data_vars}
        ds.to_netcdf(out_dir / 'grid.nc', format='NETCDF4_CLASSIC', encoding=encoding)
        quicklook(ds, domain, out_dir / 'grid_quicklook.png')
        summary_file = out_dir / 'summary.json'
        existing = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        existing['grid'] = summary
        summary_file.write_text(json.dumps(existing, indent=2))
        print(f"{domain}: grid.nc {summary['xi_rho']}x{summary['eta_rho']}, "
              f"dx {summary['dx_m'][0]:.1f}-{summary['dx_m'][2]:.1f} m, "
              f"dy {summary['dy_m'][0]:.2f}-{summary['dy_m'][2]:.2f} m, "
              f"h {summary['h_m'][0]:.2f}-{summary['h_m'][1]:.2f} m, "
              f"min angle {summary['angle_min_deg']:.1f} deg, "
              f"interior area {summary['interior_area_m2']:.0f} m2, "
              f"mean depth {summary['interior_mean_depth_m']:.2f} m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    main(parser.parse_args().domains)
