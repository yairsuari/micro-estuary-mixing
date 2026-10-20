/*
** ROMS application header: Alexander / Meander / NoMeander micro-estuaries
** (ROMS_APPLICATION = MICRO_ESTUARY). Built by build_roms.sh in this folder.
**
** Grid, initial and boundary data come from NetCDF files (ROMS/Data/<domain>).
** Options that modern ROMS sets in the .in file instead of here:
**   tracer advection (Hadvection/Vadvection: HSIMT), lateral boundary
**   conditions (LBC: west open, others closed), river point sources
**   (LuvSrc/LtracerSrc), wet/dry critical depth (DCRIT).
**
** Build-time switches (MY_CPP_FLAGS, set by build_roms.sh):
**   -DUSE_GLS   GLS k-epsilon closure instead of Mellor-Yamada 2.5
**   -DUSE_DYES  passive tracers with mean water age (needs NPT = 6 in .in)
**   -DUSE_WIND_FILE  surface wind stress from the forcing file FRCNAME (wind.nc,
**               ROMS/python/roms_wind.py; run key WIND_FILE) instead of zero stress
*/

/* Grid and domain */
#define CURVGRID              /* curvilinear orthogonal grid (pygridgen) */
#define MASKING               /* land/sea masks (all water; walls are the closed boundaries) */
#define WET_DRY               /* wetting and drying, critical depth DCRIT in .in */
#define SOLVE3D

/* Momentum */
#define UV_ADV                /* advection of momentum */
#define UV_COR                /* Coriolis (negligible at this scale, kept for completeness) */
#define UV_LOGDRAG            /* logarithmic bottom drag, Zob in .in (POM z0b = 0.01 m) */
#define LIMIT_BSTRESS         /* limit bottom stress in very shallow (wet/dry) cells */
#define UV_VIS2               /* harmonic horizontal viscosity, VISC2 in .in */
#define MIX_S_UV              /* ... along s-surfaces */
#define SPLINES_VVISC         /* splines reconstruction of vertical viscosity */
#define DJ_GRADPS             /* pressure gradient for steep terrain-following grids */

/* Tracers */
#define SALINITY
#define NONLIN_EOS
#define TS_DIF2               /* harmonic horizontal diffusion, TNU2 in .in */
#define MIX_S_TS              /* ... along s-surfaces (geopotential mixing is ill-posed over the 1 cm mouth bar) */
#define SPLINES_VDIFF         /* splines reconstruction of vertical diffusion */

/* Vertical mixing: Mellor-Yamada 2.5 (as POM), or GLS k-epsilon with -DUSE_GLS */
#ifdef USE_GLS
# define GLS_MIXING
#else
# define MY25_MIXING
#endif
#if defined GLS_MIXING || defined MY25_MIXING
# define KANTHA_CLAYSON
/* N2S2_HORAVG (horizontal smoothing of N2 and S2) is NOT used: next to the open
** boundary it averages in the dry mouth-bar column, which holds sea water, and
** falsely mixes the deep column next to the bar.
** -DUSE_N2S2_HORAVG restores it for tests. */
# ifdef USE_N2S2_HORAVG
#  define N2S2_HORAVG
# endif
# define RI_SPLINES
#endif

/* Passive tracers (NPT = 6 in .in, three mean-age pairs, see ROMS/python/common.py):
**   dye_01 / dye_01_age   river (fresh) water fraction and its age
**   dye_02 / dye_02_age   seawater fraction and its age
**   dye_03 / dye_03_age   unit tracer and ventilation age (last pair), reset to zero in the
**                         surface layer (AGE_SURFACE_RESET: patches/age_surface_reset.patch,
**                         applied to the ROMS source by build_roms.sh --dyes) */
#ifdef USE_DYES
# define T_PASSIVE
# define AGE_MEAN
# define AGE_SURFACE_RESET
#endif

/* Surface and bottom fluxes: none (no wind, no heat or salt flux) */
#ifndef USE_WIND_FILE
# define ANA_SMFLUX           /* zero wind stress, as in POM (default); with -DUSE_WIND_FILE the
                                 stress is read from FRCNAME (ROMS/python/roms_wind.py) */
#endif
#define ANA_STFLUX
#define ANA_SSFLUX
#define ANA_BTFLUX
#define ANA_BSFLUX
#ifdef T_PASSIVE
# define ANA_SPFLUX
# define ANA_BPFLUX
#endif

/* Output */
#define AVERAGES
#define DIAGNOSTICS_TS
