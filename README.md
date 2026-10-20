# What mixes the bottom water of a micro-estuary?

Model configuration, scripts and processed results for the article

> Suari, Y. *What mixes the bottom water of a micro-estuary? Model of river flow, wind and estuary shape.*
> Submitted to *Estuaries and Coasts*.

The study combines surface and bottom salinity records (2014–2025) from the Alexander micro-estuary
(southeastern Mediterranean coast) with three-dimensional simulations in the Regional Ocean Modeling System
(ROMS) of the Alexander and of two synthetic estuaries of the same area and depth, one meandering and one
straight. It examines how river flow, the wind and its daily cycle, and the shape of the estuary control the
vertical exchange of the stratified bottom salt pool. Details of the model configuration and the values behind
every figure are also given in Online Resource 1 of the article.

## Contents

```
GRIDATA/                  model domains and grids
  domains/                  Alexander_ITM.csv (digitised banks), Alexanderwith_l.csv (centreline bathymetry),
                            TS_A4.xlsx (measured salinity profile for the initial conditions), and the domain
                            polygons made from them (Alexander.csv, Meander.csv, NoMeander.csv, domain_summary.csv)
  make_domains.py           the three domain polygons (Alexander, Meander, NoMeander)
  grid_IC.py                curvilinear-orthogonal grids (pygridgen) and bathymetry -> grids/<domain>_grid.npz
  add_sea_mouth.py          sea cells and the raised sandbar crest at the mouth -> grids_bar/<domain>_grid.npz
  grid_quality.py           orthogonality and depth checks of the grids
ROMS/
  Apps/estuary/             ROMS application: CPP options (micro_estuary.h), .in template and its generator,
                            build script, and the one change to the ROMS source (patches/: the ventilation age is
                            reset to zero in the surface layer)
  setup/install_ubuntu.sh   packages, ROMS source (~/roms/src) and executables (~/roms/bin) on Ubuntu / WSL
  python/                   input files and analysis
    common.py                 paths, grid loading, staggering helpers
    roms_grid.py, roms_vgrid.py, roms_ini.py, roms_bry.py, roms_rivers.py, roms_wind.py
                              grid.nc, s-coordinates (vgrid.yaml), initial fields, sea boundary, river point
                              sources, wind (constant or diurnal sine)
    make_inputs.py            all of the above (ROMS_DATA=Data_bar selects the input folder)
    ini_from_run.py           ini.nc: the filled initial state, a record of the filling run (ini_fill.cfg)
    mixing_periods.py         fresh-water-only periods in the salinity records and their exchange rates
    salinity_spectra.py       spectra and mean daily cycles of the observed salinity (Fig. 3)
    manuscript_figures.py     Figs. 1 and 2
    compare_e1.py             base-flow runs (background diffusivity, river and wind switched on and off)
    compare_wind_iw.py        wind-variability experiment (Fig. 4)
    compare_flow.py           river-flow experiment, salinity-variance budget, Froude and Richardson numbers (Fig. 5)
    compare_delta.py          10- versus 20-layer test
    analyse_runs.py, calibrate_mixing.py, compare_layers.py   shared analysis functions
  scripts/
    defaults.cfg              default run parameters
    fill_in.py                fills the .in template for one run
    run_one.sh                one run:       run_one.sh [--dry-run] DOMAIN EXPERIMENT RUN [KEY=VALUE ...]
    run_experiment.sh         a set of runs: run_experiment.sh [--dry-run] experiments/NAME.cfg
    experiments/              the runs of the article (below)
  Data_bar/, Data_N10u/, Data_N20u/   summaries of the inputs (the NetCDF inputs are regenerated)
results/                  processed results behind the figures and tables (below)
docs/figures/             output folder of the figure scripts
```

## Reproducing the results

1. **Grids** (Python with pygridgen and geopandas), in `GRIDATA/`: `python make_domains.py`,
   `python grid_IC.py`, `python add_sea_mouth.py`.
2. **ROMS inputs**, in `ROMS/python/`: `ROMS_DATA=Data_bar python make_inputs.py --tracers`.
3. **ROMS**: `ROMS/setup/install_ubuntu.sh` downloads and builds ROMS ([myroms.org](https://www.myroms.org); the
   source is not redistributed here) with `ROMS/Apps/estuary/build_roms.sh`, which applies the patch.
4. **Runs**: `ROMS/scripts/run_experiment.sh ROMS/scripts/experiments/<name>.cfg`, in this order:
   - `ini_fill.cfg`, the 28-h filling run. Then
     `ROMS_DATA=Data_bar python ini_from_run.py --runs-dir ../../../ROMS_runs/ini_fill --day <day>` writes the
     initial state of all experiments;
   - `wind_iw_bar.cfg`, the wind-variability experiment and the base-flow runs (including background
     diffusivities of 10⁻⁷ and 10⁻⁶ m² s⁻¹ and the runs without river flow);
   - `flow_bar.cfg` and `flow_bar_fix.cfg`, the river-flow experiment (the latter reruns six runs with a 1 s time
     step);
   - `vertical_layers_20d.cfg`, the 10- versus 20-layer test (inputs from `make_inputs.py` with
     `ROMS_DATA=Data_N10u` and `Data_N20u`).

   Model output is written outside the repository, to `../ROMS_runs/<experiment>/<run>/` (override with
   `ROMS_RUNS`), with copies of the inputs, `roms.in`, `run_meta.json` (all parameters) and `roms.log`. It is not
   archived here because of its size. Run keys (see `ROMS/scripts/defaults.cfg`): `DATA` (input folder),
   `INI_FILE`, `RIVER_SCALE`, `WIND_FILE`, `AKS_BAK`, `AKT_BAK`, `AKV_BAK`, `DT`, `DAYS`.
5. **Analysis and figures**: the scripts in `ROMS/python/` read the model output and write the tables in
   `results/` and the figures (`manuscript_figures.py` needs `ROMS_GRIDS=grids_bar ROMS_DATA=Data_bar`).

The observed salinity records are not included. They are available from the Israeli estuaries research website,
[www.estuaries.org.il](https://www.estuaries.org.il) (continuous measurement data), and the database is described
in Suari et al. (2019, *Data in Brief* 27, 104809, https://doi.org/10.1016/j.dib.2019.104809).
`mixing_periods.py` expects them as `A4_salinity_2014-2025.csv` and `A3_salinity_2016-2025.csv` (date/time,
surface salinity, bottom salinity) in `../ROMS_runs/mixing_calibration/input/`.

## Grid conventions

- xi runs along the channel, with xi = 0 at the western (sea) boundary. Two rows of sea cells 2 m deep lie seaward
  of the mouth row, the sandbar crest, 0.10 m above sea level. The river end is closed, and the river enters as
  point sources through the wall.
- eta runs across the channel. The outer rows are the closed banks, so the computed channel is 5 cells wide.
- `h` is positive down. With wetting and drying (critical depth 0.05 m) the crest stays dry at sea level 0.
- Time: seconds since 2000-01-01.

## Processed results

Salinity on the Practical Salinity Scale (PSU). Exchange and loss rates in % of the salt pool per day. A4, the
station ~400 m from the mouth (Michmoret Bridge); A3, ~3500 m from the mouth (Turtle Bridge).

| File | Script | Content |
|---|---|---|
| `results/model/base_flow.csv` | `compare_e1.py` | base-flow runs of the Alexander: exchange rates at A4 and A3, salt loss, numerical share of the mixing |
| `results/model/wind_variability.csv` | `compare_wind_iw.py` | 33 runs (3 domains; no wind, constant and diurnal wind, M = 1–8 m/s): exchange rates, salt loss, halocline depth and its daily range, salinity cycles |
| `results/model/wind_daily_cycle_timing.csv` | `compare_wind_iw.py` | diurnal-wind runs: local hour of the maximum and minimum and daily range of the surface and bottom salinity ~3500 m from the mouth |
| `results/model/river_flow.csv` | `compare_flow.py` | 33 runs (3 domains; no river and 0.25–128 × 20 000 m³/day): exchange rates, bottom-pool loss rate, T50, T90, salinity-variance budget, Froude number, share of halocline cells with Ri < 0.25, ventilation age |
| `results/model/river_flow_scaling.csv` | `compare_flow.py` | power-law fits of the rates against the flow, per domain |
| `results/observations/periods.csv` | `mixing_periods.py` | fresh-water-only periods: dates, bottom-salinity slope, bottom − surface difference, exchange rate with 95% interval |
| `results/observations/daily_exchange.csv` | `mixing_periods.py` | daily exchange rates of the bottom salt pool in these periods |
| `results/observations/salinity_spectra.csv` | `salinity_spectra.py` | power spectra of the surface and bottom salinity at A4 and A3, dry and wet season |
| `results/observations/salinity_diurnal.csv` | `salinity_spectra.py` | strength of the diurnal cycle (variance share, peak to background) |
| `results/observations/salinity_daily_cycle.csv` | `salinity_spectra.py` | mean daily cycles with 90% confidence intervals |

## Requirements

Python 3.10+ with the packages in `requirements.txt`; ROMS (svn trunk, built with gfortran and NetCDF-Fortran) on
Linux or WSL.

## License

Code: MIT license (`LICENSE`). Processed results in `results/` and the domain data in `GRIDATA/domains/`: Creative
Commons Attribution 4.0 (CC BY 4.0).

## Citation

Please cite the article (details will be added on publication) and this repository (`CITATION.cff`).
