"""
Build all ROMS input files for the selected domains and suggest time steps.

Runs, in order: roms_vgrid (Data/vgrid.yaml), roms_grid, roms_ini, roms_bry,
roms_rivers, then writes a 'timestep' entry in Data/<domain>/summary.json:
    DTFAST_max = 0.8 x the barotropic CFL time step of the grid (deepest,
                 narrowest cell, with the extra sea level --zeta-max)
    DT         = largest of 1,2,3,4,5,6,10,12,15,20,30,60 s for which a current
                 of --umax m/s crosses at most half of the shortest cell, capped at
                 --dt-max (default 2 s x bottom layer fraction / 0.04, at most 2 s). The cap is the vertical-advection limit
                 found in the first runs: the initial salinity front slumps fastest
                 at the deepest cells next to the mouth bar, where the bottom layers
                 are thin; DT = 6 s blew up there (vertical Courant number > 0.3),
                 DT = 2 s is stable (vertical Courant number <= 0.14).
    NDTFAST    = ceil(DT / DTFAST_max)

The grids themselves come from GRIDATA (pygridgen, WSL env gridding4pom):
    cd GRIDATA; python make_domains.py; python grid_IC.py

Default vertical grid: 10 layers, mild surface refinement (roms_vgrid.DEFAULT).
Usage: ROMS_DATA=Data_N20u python make_inputs.py --tracers --n 20 --vstretching 1 --theta-s 0  (uniform)
       python make_inputs.py [--domains ...] [--river-scale 1] [--sea-salt 39.5]
                             [--zeta-csv file] [--tracers]
"""
import argparse
import json
import math

import numpy as np
import xarray as xr

import roms_bry
import roms_grid
import roms_ini
import roms_rivers
import roms_vgrid
from common import DOMAINS, domain_dir

G = 9.81
DT_CHOICES = (0.5, 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)


def suggest_timestep(domain, umax=1.0, zeta_max=0.5, courant=0.8, adv_fraction=0.5, dt_max=2.0):
    with xr.open_dataset(domain_dir(domain) / 'grid.nc') as gr:
        dx = 1.0 / gr.pm.values[1:-1, 1:-1]
        dy = 1.0 / gr.pn.values[1:-1, 1:-1]
        h = gr.h.values[1:-1, 1:-1]
    depth = np.maximum(h, 0.05) + zeta_max
    dt_cfl = 1.0 / (np.sqrt(G * depth) * np.sqrt(1 / dx ** 2 + 1 / dy ** 2))
    dtfast_max = courant * dt_cfl.min()
    dt_adv = adv_fraction * dx.min() / umax
    dt = max([d for d in DT_CHOICES if d <= min(dt_adv, dt_max)] or [DT_CHOICES[0]])
    ndtfast = math.ceil(dt / dtfast_max)
    return {'DT': float(dt), 'NDTFAST': int(ndtfast), 'DTFAST': dt / ndtfast,
            'DTFAST_max': float(dtfast_max), 'barotropic_cfl_s': float(dt_cfl.min()),
            'advective_limit_s': float(dt_adv), 'dt_max_s': float(dt_max),
            'umax_m_s': umax, 'zeta_max_m': zeta_max}


def main(domains=DOMAINS, river_scale=1.0, sea_salt=roms_bry.SEA_SALT, salt_profile='sea',
         zeta_csv=None, dcrit=0.05, tracers=False, umax=1.0, zeta_max=0.5,
         dt_max=None, n_layers=roms_vgrid.DEFAULT['N'], vstretching=roms_vgrid.DEFAULT['Vstretching'],
         theta_s=roms_vgrid.DEFAULT['theta_s'], theta_b=roms_vgrid.DEFAULT['theta_b']):
    p = roms_vgrid.main(N=n_layers, vstretching=vstretching, theta_s=theta_s, theta_b=theta_b)
    if dt_max is None:
        # 2 s was found stable with the POM-fitted N = 7 levels, whose bottom layer is 4% of the
        # depth (the instability was at the bottom next to the bar); scale with the bottom layer,
        # never above 2 s
        dt_max = min(2.0, 2.0 * p['bottom_layer_fraction'] / 0.040)
    roms_grid.main(domains)
    roms_ini.main(domains, dcrit, tracers)
    roms_bry.main(domains, salt_profile, sea_salt, zeta_csv, dcrit, tracers)
    roms_rivers.main(domains, roms_rivers.RIVER_FLOW, river_scale, tracers)
    for domain in domains:
        ts = suggest_timestep(domain, umax, zeta_max, dt_max=dt_max)
        summary_file = domain_dir(domain) / 'summary.json'
        summary = json.loads(summary_file.read_text())
        summary['timestep'] = ts
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"{domain}: DT {ts['DT']:g} s, NDTFAST {ts['NDTFAST']} (DTFAST {ts['DTFAST']:.3f} s; "
              f"barotropic CFL {ts['barotropic_cfl_s']:.3f} s, advective limit {ts['advective_limit_s']:.1f} s)")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build all ROMS input files')
    p.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument('--river-scale', type=float, default=1.0)
    p.add_argument('--sea-salt', type=float, default=roms_bry.SEA_SALT)
    p.add_argument('--salt-profile', choices=['sea', 'ic'], default='sea')
    p.add_argument('--zeta-csv', default=None)
    p.add_argument('--dcrit', type=float, default=0.05)
    p.add_argument('--tracers', action='store_true',
                   help='passive tracers: river water fraction and ventilation age (common.PASSIVE)')
    p.add_argument('--umax', type=float, default=1.0, help='velocity scale for the advective time-step limit (m/s)')
    p.add_argument('--zeta-max', type=float, default=0.5, help='sea level margin for the barotropic limit (m)')
    p.add_argument('--dt-max', type=float, default=None,
                   help='cap on DT (s), vertical-advection limit (default 2 s x thinnest layer / 0.04, max 2 s)')
    p.add_argument('--n', type=int, default=roms_vgrid.DEFAULT['N'], help='number of vertical layers (default 10)')
    p.add_argument('--vstretching', type=int, default=roms_vgrid.DEFAULT['Vstretching'], choices=[1, 4])
    p.add_argument('--theta-s', type=float, default=roms_vgrid.DEFAULT['theta_s'])
    p.add_argument('--theta-b', type=float, default=roms_vgrid.DEFAULT['theta_b'])
    p.add_argument('--pom-fit', action='store_true', help='the original levels fitted to POM (N = 7)')
    a = p.parse_args()
    main(a.domains, a.river_scale, a.sea_salt, a.salt_profile, a.zeta_csv, a.dcrit,
         a.tracers, a.umax, a.zeta_max, a.dt_max, 7 if a.pom_fit else a.n, None if a.pom_fit else a.vstretching,
         a.theta_s, a.theta_b)
