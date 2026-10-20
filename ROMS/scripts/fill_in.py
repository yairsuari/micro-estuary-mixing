"""
Fill the ROMS input template for one run (standard library only, so it runs
with the system python3 in WSL/Ubuntu).

    python3 fill_in.py DOMAIN EXPERIMENT RUN OUTDIR [KEY=VALUE ...]

Parameter precedence (last wins):
    ROMS/scripts/defaults.cfg
    DT / NDTFAST suggested in ROMS/Data/<DOMAIN>/summary.json
    KEY=VALUE arguments
Writes OUTDIR/roms.in and OUTDIR/run_meta.json and prints the executable
name and number of processes for run_one.sh.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROMS_DIR = SCRIPTS.parent
TEMPLATE = ROMS_DIR / 'Apps' / 'estuary' / 'roms_micro_estuary.in.tmpl'


def read_cfg(path):
    params = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split('=', 1)
            params[key.strip()] = value.strip()
    return params


def read_flat_yaml(path):
    '''Minimal reader for the flat "key: value" file written by roms_vgrid.py.'''
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if ':' in line:
            key, value = (s.strip() for s in line.split(':', 1))
            out[key] = value
    return out


def fortran_real(value):
    '''1.5 -> 1.5d0, 3.162e-7 -> 3.162d-7 (ROMS reads both, d keeps it double).'''
    v = float(value)
    s = repr(v)
    if 'e' in s:
        mant, exp = s.split('e')
        return f'{mant}d{int(exp)}'
    return f'{s}d0'


def steps(seconds, dt):
    n = round(float(seconds) / dt)
    if n < 1:
        raise ValueError(f'interval of {seconds} s is shorter than DT = {dt} s')
    return n


# .in keys with one value per tracer (NAT active + NPT passive). The template
# lists temperature and salinity only; extend_tracer_lists() appends the
# passive tracers with the salinity setting (same advection, diffusivity,
# nudging, boundary types, river switch).
TRACER_KEYS = ('TNU2', 'TNU4', 'ad_TNU2', 'ad_TNU4', 'LtracerSponge', 'AKT_BAK', 'ad_AKT_fac', 'TNUDG',
               'LtracerSrc', 'LtracerCLM', 'LnudgeTCLM', 'Lstate(isTvar)', 'Fstate(isTvar)', 'Fstate(isTsur)')
TRACER_BLOCKS = ('Hadvection', 'Vadvection', 'ad_Hadvection', 'ad_Vadvection', 'LBC(isTvar)', 'ad_LBC(isTvar)')
PASSIVE_NAMES = ('dye_01', 'dye_01_age', 'dye_02', 'dye_02_age', 'dye_03', 'dye_03_age')   # common.py PASSIVE
BACKSLASH = chr(92)                                                     # .in line continuation


def _split(line):
    body, _, comment = line.partition('!')
    return body.rstrip(), ('!' + comment) if comment else ''


def extend_tracer_lists(text, npt):
    if npt == 0:
        return text
    names = [PASSIVE_NAMES[i] if i < len(PASSIVE_NAMES) else f'inert {i + 1}' for i in range(npt)]
    lines = text.split('\n')
    out = []
    k = 0
    while k < len(lines):
        line = lines[k]
        key = line.split('==')[0].strip() if '==' in line and not line.lstrip().startswith('!') else None
        if key in TRACER_KEYS or (key or '').startswith('Dout(iT'):
            body, comment = _split(line)
            head, values = body.split('==', 1)
            tokens = []
            for tok in values.split():
                if '*' in tok:
                    n, v = tok.split('*', 1)
                    tokens += [v] * int(n)
                else:
                    tokens.append(tok)
            tokens += [tokens[-1]] * npt
            out.append(f'{head}== {" ".join(tokens)}  {comment}'.rstrip())
            k += 1
        elif key in TRACER_BLOCKS:
            block = [line]
            while _split(block[-1])[0].endswith(BACKSLASH):
                k += 1
                block.append(lines[k])
            last_body, last_comment = _split(block[-1])
            value = last_body.strip()                      # salinity entry
            indent = ' ' * (len(last_body) - len(last_body.lstrip()))
            block[-1] = f'{last_body} {BACKSLASH}  {last_comment}'.rstrip()
            for i, name in enumerate(names):
                cont = ' ' + BACKSLASH if i < npt - 1 else ''
                block.append(f'{indent}{value}{cont}  ! {name}')
            out += block
            k += 1
        else:
            out.append(line)
            k += 1
    return '\n'.join(out)


def main(domain, experiment, run, outdir, overrides):
    outdir = Path(outdir)
    params = read_cfg(SCRIPTS / 'defaults.cfg')
    for item in overrides:                       # DATA is needed before the rest
        if item.startswith('DATA='):
            params['DATA'] = item.split('=', 1)[1]
    DATA = ROMS_DIR / params['DATA']
    summary = json.loads((DATA / domain / 'summary.json').read_text())
    params['DT'] = str(summary['timestep']['DT'])
    params['NDTFAST'] = str(summary['timestep']['NDTFAST'])
    for item in overrides:
        key, value = item.split('=', 1)
        params[key] = value
    vg = read_flat_yaml(DATA / 'vgrid.yaml')

    dt = float(params['DT'])
    rivers = params['RIVERS'].lower() in ('on', 'yes', 't', 'true', '1')
    values = {
        'DOMAIN': domain, 'EXPERIMENT': experiment, 'RUN': run,
        'VARINFO': os.environ.get('ROMS_VARINFO',
                                  str(Path(os.environ.get('ROMS_SRC', Path.home() / 'roms/src'))
                                      / 'ROMS/External/varinfo.yaml')),
        'LM': summary['grid']['xi_rho'] - 2,
        'MM': summary['grid']['eta_rho'] - 2,
        'N': vg['N'], 'NPT': params['NPT'],
        'NTILEI': params['NTILEI'], 'NTILEJ': params['NTILEJ'],
        'DT': fortran_real(dt), 'NDTFAST': int(params['NDTFAST']),
        'NTIMES': steps(float(params['DAYS']) * 86400, dt),
        'NHIS': steps(float(params['HIS_HOURS']) * 3600, dt),
        'NAVG': steps(float(params['AVG_HOURS']) * 3600, dt),
        'NDIA': steps(float(params['DIA_HOURS']) * 3600, dt),
        'NRST': steps(float(params['RST_DAYS']) * 86400, dt),
        'NINFO': steps(float(params['INFO_MINUTES']) * 60, dt),
        'TNU2': fortran_real(params['TNU2']), 'VISC2': fortran_real(params['VISC2']),
        'AKT_BAK': fortran_real(params['AKT_BAK']), 'AKS_BAK': fortran_real(params['AKS_BAK']),
        'AKV_BAK': fortran_real(params['AKV_BAK']),
        'ZOB': fortran_real(params['ZOB']), 'DCRIT': fortran_real(params['DCRIT']),
        'VTRANSFORM': vg['Vtransform'], 'VSTRETCHING': vg['Vstretching'],
        'THETA_S': fortran_real(vg['theta_s']), 'THETA_B': fortran_real(vg['theta_b']),
        'TCLINE': fortran_real(vg['Tcline']),
        'TIME_REF': params['TIME_REF'],
        'TNUDG': fortran_real(params['TNUDG']), 'OBCFAC': fortran_real(params['OBCFAC']),
        'LUVSRC': 'T' if rivers else 'F', 'LTRACERSRC': 'T' if rivers else 'F',
        'GRDNAME': 'grid.nc', 'ININAME': 'ini.nc', 'BRYNAME': 'bry.nc', 'SSFNAME': 'rivers.nc',
        'FRCNAME': 'wind.nc',          # read only by builds with -DUSE_WIND_FILE (run key WIND_FILE)
        'OUTDIR': '.',
    }
    text = TEMPLATE.read_text()
    text = re.sub(r'@([A-Z_0-9]+)@', lambda m: str(values[m.group(1)]), text)
    text = extend_tracer_lists(text, int(params['NPT']))
    left = re.findall(r'@[A-Z_0-9]+@', text)
    if left:
        raise ValueError(f'unfilled placeholders: {left}')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'roms.in').write_text(text, newline='\n')

    try:
        commit = subprocess.run(['git', '-C', str(ROMS_DIR), 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ''
    meta = {'domain': domain, 'experiment': experiment, 'run': run,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_commit': commit, 'parameters': params,
            'filled': {k: str(v) for k, v in values.items()},
            'vgrid': vg, 'input_summary': summary}
    (outdir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    print(params['EXEC'], params['NPROC'], 'rivers' if rivers else 'norivers')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
