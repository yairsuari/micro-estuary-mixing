"""
E1 on the closed mouth: the mixing of the salt pool at base flow split by forcing. Alexander, runs of ROMS_runs/<experiment> (wind_iw_bar), 30 days, sea level 0:
    static   e1_static   no river, no wind (the adjustment of the initial state and numerical mixing)
    wind     e1_wind     constant wind 2 m/s only
    river    none        river 20 000 m3/day only
    both     const_2     river and constant wind 2 m/s
    both, AKS = AKT = 1e-7 and 1e-6 m2/s (e1_aks1e7, e1_aks1e6): sensitivity to the background diffusivity
Per run:
    A4, A3       mean daily exchange rate at the sensor positions (calibrate_mixing.daily_exchange, days FIRST-LAST,
                 days with S_b - S_s >= DS_MIN, as in the wind-variability experiment), and the number of days
    pool rate    loss rate of the bottom pool (compare_flow: salt deeper than POOL_DEPTH below the initial surface)
    num share    numerical share of the mixing (compare_flow: salinity-square budget)
and the split: the forcing contributions above the static floor, wind and river alone and together, and the
interaction (both - wind - river + static).

Writes ROMS_runs/<experiment>/output_figures/e1.csv and prints the table and the split.

Usage: python compare_e1.py [--experiment wind_iw_bar]
"""
import argparse

import numpy as np
import pandas as pd

import calibrate_mixing as cm
import compare_flow as cf

RUNS = {'static': ('e1_static', 0.0), 'wind': ('e1_wind', 0.0), 'river': ('none', 1.0), 'both': ('const_2', 1.0),
        'both_aks1e7': ('e1_aks1e7', 1.0), 'both_aks1e6': ('e1_aks1e6', 1.0)}
FIRST, LAST = 1, 30
DS_MIN = 3.0


def daily_rate(s, st):
    cm.FIRST_DAY, cm.LAST_DAY = FIRST, min(LAST, int(s.day.iloc[-1]))
    d = cm.daily_exchange(s, st)
    d = d[d.dS >= DS_MIN]
    return d.exchange.mean(), len(d)


def main(experiment='wind_iw_bar'):
    cf.set_experiment(experiment)
    rows = []
    for case, (name, scale) in RUNS.items():
        path = cf.EXP / name
        if not cf.finished(path):
            print(f'{name}: not finished, skipped')
            continue
        q = scale * cf.Q_REF / 86400.0
        r = cf.cached(path, q)
        summ = cf.summarise(r, q)
        row = {'case': case, 'run': name}
        for st in ('A4', 'A3'):
            row[f'{st}_exchange'], row[f'{st}_days'] = daily_rate(r['series'], st)
        row.update({k: summ[k] for k in ('pool_rate', 'T50', 'num_share', 'chi_phys', 'chi_num',
                                          'halocline_AKs_mean')})
        rows.append(row)
    if not rows:
        raise SystemExit(f'no finished E1 runs in {cf.EXP}')
    t = pd.DataFrame(rows).set_index('case')
    cf.OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(cf.OUT / 'e1.csv', float_format='%.4g')
    with pd.option_context('display.width', 200):
        print(t.to_string(float_format=lambda v: f'{v:.3g}'))
    if {'static', 'wind', 'river', 'both'} <= set(t.index):
        for col in ('A4_exchange', 'A3_exchange', 'pool_rate'):
            v = t[col]
            print(f'{col}: floor {v.static:.3g}, wind +{v.wind - v.static:.3g}, river +{v.river - v.static:.3g}, '
                  f'both +{v.both - v.static:.3g}, interaction {v.both - v.wind - v.river + v.static:+.3g} (%/day)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='wind_iw_bar')
    main(ap.parse_args().experiment)
