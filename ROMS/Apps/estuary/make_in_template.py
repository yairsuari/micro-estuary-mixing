"""
Create roms_micro_estuary.in.tmpl from the standard ROMS input file
ROMS/External/roms_upwelling.in of the ROMS source tree.

Only the settings listed in CHANGES / BLOCKS below differ from the upstream
file, so this script is also the record of what was changed. Values in
@...@ are placeholders filled per run by ROMS/scripts/run_one.sh (from
ROMS/Data/vgrid.yaml, ROMS/Data/<domain>/summary.json and the experiment cfg).

Usage: python make_in_template.py [path/to/roms_upwelling.in]
       (default: $HOME/roms/src/ROMS/External/roms_upwelling.in)
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'roms_micro_estuary.in.tmpl'

# key -> new value (first non-adjoint occurrence of "key == value" or "key = value")
CHANGES = {
    'TITLE': 'Micro-estuary @DOMAIN@ (@EXPERIMENT@/@RUN@)',
    'MyAppCPP': 'MICRO_ESTUARY',
    'VARNAME': '@VARINFO@',
    'Lm': '@LM@',
    'Mm': '@MM@',
    'N': '@N@',
    'NPT': '@NPT@',
    'NtileI': '@NTILEI@',
    'NtileJ': '@NTILEJ@',
    'NTIMES': '@NTIMES@',
    'DT': '@DT@',
    'NDTFAST': '@NDTFAST@',
    'NRREC': '0',
    'NRST': '@NRST@',
    'NINFO': '@NINFO@',
    'NHIS': '@NHIS@',
    'NDEFHIS': '0',
    'NAVG': '@NAVG@',
    'NDIA': '@NDIA@',
    'TNU2': '@TNU2@ @TNU2@',
    'VISC2': '@VISC2@',
    'AKT_BAK': '@AKT_BAK@ @AKS_BAK@',
    'AKV_BAK': '@AKV_BAK@',
    'Zob': '@ZOB@',
    'DCRIT': '@DCRIT@',
    'Vtransform': '@VTRANSFORM@',
    'Vstretching': '@VSTRETCHING@',
    'THETA_S': '@THETA_S@',
    'THETA_B': '@THETA_B@',
    'TCLINE': '@TCLINE@',
    'TIME_REF': '@TIME_REF@',
    'TNUDG': '2*@TNUDG@',
    'OBCFAC': '@OBCFAC@',
    'LuvSrc': '@LUVSRC@',
    'LtracerSrc': '@LTRACERSRC@ @LTRACERSRC@',
    'GRDNAME': '@GRDNAME@',
    'ININAME': '@ININAME@',
    'BRYNAME': '@BRYNAME@',
    'SSFNAME': '@SSFNAME@',
    'FRCNAME': '@FRCNAME@',
    'Hout(idVvis)': 'T', 'Hout(idTdif)': 'T', 'Hout(idSdif)': 'T',
    'Hout(idMtke)': 'T', 'Hout(idMtls)': 'T',
    'Aout(idVvis)': 'T', 'Aout(idTdif)': 'T', 'Aout(idSdif)': 'T',
}
# output files go to the run folder
for name in ('GST', 'RST', 'HIS', 'XTR', 'QCK', 'TLM', 'TLF', 'ADJ', 'AVG', 'HAR', 'DIA', 'STA', 'FLT'):
    CHANGES[f'{name}NAME'] = f'@OUTDIR@/roms_{name.lower()}.nc'

# multi-line blocks (exact upstream text -> new text)
# West sea level: clamped (Cla) to bry.nc. The boundary rho point is the 1 cm mouth bar;
# Chapman (Cha) only radiates and never imposes the boundary sea level, so a raised sea
# level could not wet the bar (seawater-dye test).
BLOCKS = [
    ("""   Hadvection == U3       \\                     ! temperature
                 HSIMT                          ! salinity""",
     """   Hadvection == HSIMT    \\                     ! temperature
                 HSIMT                          ! salinity"""),
    ("""   Vadvection == C4       \\                     ! temperature
                 HSIMT                          ! salinity""",
     """   Vadvection == HSIMT    \\                     ! temperature
                 HSIMT                          ! salinity"""),
    ("""   LBC(isFsur) ==   Per     Clo     Per     Clo         ! free-surface
   LBC(isUbar) ==   Per     Clo     Per     Clo         ! 2D U-momentum
   LBC(isVbar) ==   Per     Clo     Per     Clo         ! 2D V-momentum
   LBC(isUvel) ==   Per     Clo     Per     Clo         ! 3D U-momentum
   LBC(isVvel) ==   Per     Clo     Per     Clo         ! 3D V-momentum
   LBC(isMtke) ==   Per     Clo     Per     Clo         ! mixing TKE

   LBC(isTvar) ==   Per     Clo     Per     Clo \\       ! temperature
                    Per     Clo     Per     Clo         ! salinity""",
     """   LBC(isFsur) ==   Cla     Clo     Clo     Clo         ! free-surface
   LBC(isUbar) ==   Fla     Clo     Clo     Clo         ! 2D U-momentum
   LBC(isVbar) ==   Gra     Clo     Clo     Clo         ! 2D V-momentum
   LBC(isUvel) ==   Rad     Clo     Clo     Clo         ! 3D U-momentum
   LBC(isVvel) ==   Gra     Clo     Clo     Clo         ! 3D V-momentum
   LBC(isMtke) ==   Gra     Clo     Clo     Clo         ! mixing TKE

   LBC(isTvar) ==   RadNud  Clo     Clo     Clo \\       ! temperature
                    RadNud  Clo     Clo     Clo         ! salinity"""),
    ("""   LBC(isU2Sd) ==   Per     Clo     Per     Clo         ! 2D U-Stokes
   LBC(isV2Sd) ==   Per     Clo     Per     Clo         ! 2D V-Stokes
   LBC(isU3Sd) ==   Per     Clo     Per     Clo         ! 3D U-Stokes
   LBC(isV3Sd) ==   Per     Clo     Per     Clo         ! 3D V-Stokes""",
     """   LBC(isU2Sd) ==   Clo     Clo     Clo     Clo         ! 2D U-Stokes
   LBC(isV2Sd) ==   Clo     Clo     Clo     Clo         ! 2D V-Stokes
   LBC(isU3Sd) ==   Clo     Clo     Clo     Clo         ! 3D U-Stokes
   LBC(isV3Sd) ==   Clo     Clo     Clo     Clo         ! 3D V-Stokes"""),
]


def set_value(lines, key, value):
    '''Replace the value of the first "key == ..." / "key = ..." line (not ad_*).'''
    pat = re.compile(r'^(\s*)' + re.escape(key) + r'(\s*==?\s*)(.*?)(\s+!.*)?$')
    for k, line in enumerate(lines):
        m = pat.match(line)
        if m:
            comment = m.group(4) or ''
            lines[k] = f'{m.group(1)}{key}{m.group(2)}{value}{comment}'
            return
    raise KeyError(f'{key} not found in the upstream input file')


def main(src):
    text = Path(src).read_text()
    for old, new in BLOCKS:
        if old not in text:
            raise ValueError(f'block not found upstream:\n{old[:80]}')
        text = text.replace(old, new, 1)
    lines = text.split('\n')
    for key, value in CHANGES.items():
        set_value(lines, key, value)
    header = ('! ROMS input template for the micro-estuary simulations, generated by\n'
              f'! ROMS/Apps/estuary/make_in_template.py from {Path(src).name} of the ROMS source.\n'
              '! @...@ placeholders are filled by ROMS/scripts/run_one.sh. Do not edit by hand:\n'
              '! change make_in_template.py and regenerate.\n!\n')
    OUT.write_text(header + '\n'.join(lines), newline='\n')
    left = sorted(set(re.findall(r'@[A-Z_0-9]+@', OUT.read_text())))
    print(f'wrote {OUT}\nplaceholders: {" ".join(left)}')


if __name__ == '__main__':
    default = Path(os.environ.get('HOME', '~')) / 'roms/src/ROMS/External/roms_upwelling.in'
    main(sys.argv[1] if len(sys.argv) > 1 else default)
