"""
Build the three model domains (Alexander, Meander, NoMeander) as boundary
polygons for pygridgen, replacing the construction in Alexander_2_domains.xlsx.

Source: domains/Alexander_ITM.csv, a manual digitisation of the Alexander
estuary in ITM (EPSG:6984). The polygon runs
    id1 (mouth, bank B) -> id2 (mouth, bank A) -> ids 3..28 (bank A)
    -> id29 (river end, bank A) -> id30 (river end, bank B) -> ids 31..56 (bank B)
and vertex id k on bank A faces vertex id 59-k on bank B.

Outputs (columns ID,beta,x,y; beta=1 at the four corners):
    domains/Alexander.csv   - Alexander with the two end edges squared and the
                              banks lightly smoothed (densified + Gaussian
                              moving average along each bank)
    domains/NoMeander.csv   - one straight bank, the other offset by the width
    domains/Meander.csv     - sine-generated centreline, banks at +-width/2
    domains/domain_summary.csv, domains/domains_comparison.png,
    domains/alexander_smoothing.png (digitised vs smoothed banks at the sharpest bends)

All three share the Alexander centreline length and along-channel width
profile. Meander and NoMeander widths are multiplied by a single factor so
their area equals the Alexander area.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# np.trapz was renamed np.trapezoid in numpy 2.0
trapezoid = getattr(np, "trapezoid", None) or np.trapz

N_MEANDERS = 10
POINTS_PER_WAVELENGTH = 24
# Sinuosity of the previous (spreadsheet) Meander domain: centreline length
# 6690 m over a 2308 m down-valley distance between its mouth and river corners.
SINUOSITY = 2.9
# Light smoothing of the digitised Alexander banks (Gaussian standard deviation
# along the bank, m). Removes the sharp kinks of the 28-vertex digitisation that
# produced 14-degree grid cells. 0 disables smoothing.
ALEXANDER_SMOOTH_SIGMA = 30.0
DENSE_SPACING = 5.0          # m, resampling before smoothing
VERTEX_SPACING = 20.0        # m, resampling of the smoothed bank for pygridgen


def signed_area(x, y):
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_alexander(path='./domains/Alexander_ITM.csv'):
    '''
    Return the paired bank vertices of the Alexander digitisation,
    ordered from the mouth (index 0) to the river end (index -1).
    '''
    df = pd.read_csv(path, encoding='utf-8-sig')
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    # drop the closing vertex (id57 == id1)
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
        x, y = x[:-1], y[:-1]
    ids_a = np.arange(2, 30)                 # ids 2..29
    ids_b = 59 - ids_a                       # partners 57(=1), 56, ..., 30
    ids_b[ids_b == 57] = 1
    bank_a = np.c_[x[ids_a - 1], y[ids_a - 1]]
    bank_b = np.c_[x[ids_b - 1], y[ids_b - 1]]
    return bank_a, bank_b


def centreline(bank_a, bank_b):
    mid = 0.5 * (bank_a + bank_b)
    width = np.hypot(*(bank_a - bank_b).T)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(mid, axis=0).T))]
    return mid, width, s


def resample(line, spacing):
    '''Resample a polyline at (nearly) equal arc-length spacing, keeping both ends.'''
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    n = max(int(np.ceil(s[-1] / spacing)), 1) + 1
    s_new = np.linspace(0.0, s[-1], n)
    return np.c_[np.interp(s_new, s, line[:, 0]), np.interp(s_new, s, line[:, 1])]


def smooth_bank(bank, sigma, dense=DENSE_SPACING, spacing=VERTEX_SPACING):
    '''
    Densify a bank polyline and apply a Gaussian moving average along it.
    The line is extended at both ends by point reflection (2*p0 - p), which
    keeps the end points and the end directions fixed, so the squared end
    edges and the corners stay where they are.
    '''
    p = resample(bank, dense)
    if sigma > 0:
        m = int(np.ceil(3 * sigma / dense))
        m = min(m, len(p) - 1)
        k = np.arange(-m, m + 1) * dense
        w = np.exp(-0.5 * (k / sigma) ** 2)
        w /= w.sum()
        head = 2 * p[0] - p[1:m + 1][::-1]
        tail = 2 * p[-1] - p[-m - 1:-1][::-1]
        pp = np.vstack([head, p, tail])
        p = np.c_[np.convolve(pp[:, 0], w, mode='valid'),
                  np.convolve(pp[:, 1], w, mode='valid')]
    return resample(p, spacing)


def distance_to_polyline(points, line):
    '''Shortest distance from each point to a polyline.'''
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return np.hypot(*(points[:, None, :] - closest).transpose(2, 0, 1)).min(axis=1)


def turning_angles(line):
    '''Absolute change of direction (deg) at each interior vertex of a polyline.'''
    d = np.diff(line, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees(np.abs((np.diff(th) + np.pi) % (2 * np.pi) - np.pi))


def plot_alexander_smoothing(raw_a, raw_b, sm_a, sm_b, sigma, out, n_zoom=2, half=120.0):
    '''Full view plus zooms on the sharpest bends of the digitisation.'''
    cand = []
    for bank in (raw_a, raw_b):
        ang = turning_angles(bank)
        for k in np.argsort(ang)[::-1]:
            cand.append((ang[k], bank[k + 1]))
    cand.sort(key=lambda c: -c[0])
    zooms = []
    for ang, p in cand:
        if all(np.hypot(*(p - q)) > 2 * half for _, q in zooms):
            zooms.append((ang, p))
        if len(zooms) == n_zoom:
            break
    fig, axes = plt.subplots(1, n_zoom + 1, figsize=(4.2 * (n_zoom + 1), 4.4))
    for ax in axes:
        for bank in (raw_a, raw_b):
            ax.plot(bank[:, 0], bank[:, 1], '-o', color='#999999', ms=3, lw=1)
        for bank in (sm_a, sm_b):
            ax.plot(bank[:, 0], bank[:, 1], '-', color='#1f5fa8', lw=1.6)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
    axes[0].plot([], [], '-o', color='#999999', ms=3, label='digitised (squared ends)')
    axes[0].plot([], [], '-', color='#1f5fa8', label=f'smoothed (sigma {sigma:.0f} m)')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].set_title('Alexander banks', fontsize=9, loc='left')
    for k, (ang, p) in enumerate(zooms, start=1):
        axes[k].set_xlim(p[0] - half, p[0] + half)
        axes[k].set_ylim(p[1] - half, p[1] + half)
        axes[k].set_title(f'Bend {k}: {ang:.0f} deg turn in the digitisation', fontsize=9, loc='left')
        axes[0].plot(*p, 's', mfc='none', mec='#c0392b', ms=10)
        axes[0].annotate(str(k), p, xytext=(6, 6), textcoords='offset points', color='#c0392b', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def left_normal(t):
    t = t / np.linalg.norm(t)
    return np.array([-t[1], t[0]])


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def square_ends(bank_a, bank_b):
    '''
    Move the two corner vertices at each end so the closing edge is
    perpendicular to the local centreline direction (same width, same
    mid-point). Interior vertices are unchanged.
    '''
    bank_a = bank_a.copy()
    bank_b = bank_b.copy()
    mid, width, _ = centreline(bank_a, bank_b)
    for end, nxt in ((0, 1), (-1, -2)):
        t = mid[nxt] - mid[end] if end == 0 else mid[end] - mid[nxt]
        n = left_normal(t)
        # n points towards bank B if B is on the left of the flow direction
        side = np.sign(cross2(t, bank_b[end] - bank_a[end]))
        bank_b[end] = mid[end] + side * 0.5 * width[end] * n
        bank_a[end] = mid[end] - side * 0.5 * width[end] * n
    return bank_a, bank_b


def to_polygon(bank_a, bank_b):
    '''
    Assemble the pygridgen boundary in the same vertex order as the original
    digitisation: B mouth, A mouth..A river, B river..B (back towards mouth).
    '''
    # The first vertex is repeated at the end (beta=0), as in the original
    # digitisation: without it pygridgen maps the 200 points across the
    # channel instead of along it.
    pts = np.vstack([bank_b[:1], bank_a, bank_b[:0:-1], bank_b[:1]])
    beta = np.zeros(len(pts), dtype=int)
    n_a = len(bank_a)
    beta[[0, 1, n_a, n_a + 1]] = 1
    return pd.DataFrame({'ID': np.arange(1, len(pts) + 1), 'beta': beta,
                         'x': pts[:, 0], 'y': pts[:, 1]})


def polygon_area(poly):
    return abs(signed_area(poly['x'].to_numpy(), poly['y'].to_numpy()))


def self_intersects(poly):
    '''True if any two non-adjacent polygon edges cross.'''
    p = poly[['x', 'y']].to_numpy()
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    q = np.roll(p, -1, axis=0)
    n = len(p)
    d = q - p
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if j.size == 0:
            continue
        r = p[j] - p[i]
        den = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
        ok = np.abs(den) > 1e-12
        ta = (r[:, 0] * d[j, 1] - r[:, 1] * d[j, 0])[ok] / den[ok]
        tb = (r[:, 0] * d[i, 1] - r[:, 1] * d[i, 0])[ok] / den[ok]
        if np.any((ta > 0) & (ta < 1) & (tb > 0) & (tb < 1)):
            return True
    return False


def bessel_j0(w):
    tau = np.linspace(0.0, np.pi, 2001)
    return trapezoid(np.cos(w * np.sin(tau)), tau) / np.pi


def omega_for_sinuosity(sinuosity):
    '''Maximum angle (rad) of a sine-generated curve with this sinuosity: J0(w) = 1/S.'''
    target = 1.0 / sinuosity
    lo, hi = 0.0, 2.4048          # J0 decreases monotonically to its first zero
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bessel_j0(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_nomeander(origin, direction, side, s, width, scale):
    '''
    Bank A is a straight line of the centreline length starting at origin;
    bank B is offset from it by the scaled width, on the same side as in the
    Alexander digitisation.
    '''
    t = direction / np.linalg.norm(direction)
    n = side * left_normal(t)
    bank_a = origin + s[:, None] * t
    bank_b = bank_a + (scale * width)[:, None] * n
    return bank_a, bank_b


def build_meander(origin, direction, side, s_station, width_station, scale,
                  wavelength, omega):
    '''
    Sine-generated centreline (Langbein & Leopold 1966):
        theta(s) = theta0 + omega * sin(2 pi s / wavelength)
    with banks at +-width/2 along the local normal.
    '''
    length = s_station[-1]
    n_pts = int(round(length / wavelength * POINTS_PER_WAVELENGTH)) + 1
    s = np.linspace(0.0, length, n_pts)
    theta0 = np.arctan2(direction[1], direction[0])
    # integrate the heading on a fine grid, then sample
    s_fine = np.linspace(0.0, length, 50 * n_pts)
    theta = theta0 + omega * np.sin(2 * np.pi * s_fine / wavelength)
    ds = np.diff(s_fine)
    th_mid = 0.5 * (theta[1:] + theta[:-1])
    cx = np.r_[0.0, np.cumsum(ds * np.cos(th_mid))]
    cy = np.r_[0.0, np.cumsum(ds * np.sin(th_mid))]
    cx = np.interp(s, s_fine, cx) + origin[0]
    cy = np.interp(s, s_fine, cy) + origin[1]
    th = theta0 + omega * np.sin(2 * np.pi * s / wavelength)
    normal = side * np.c_[-np.sin(th), np.cos(th)]
    half_w = 0.5 * scale * np.interp(s, s_station, width_station)
    mid = np.c_[cx, cy]
    bank_a = mid - half_w[:, None] * normal
    bank_b = mid + half_w[:, None] * normal
    # minimum radius of curvature relative to the half width (inner bank)
    r_min = wavelength / (2 * np.pi * omega) if omega > 0 else np.inf
    return bank_a, bank_b, r_min, half_w.max()


def grid_area(poly, shape):
    '''
    Water area covered by the pygridgen grid of this polygon (sum of the
    quadrilateral cells). With ~20 points per meander wavelength the grid cuts
    the corners of tight bends, so it can be a few percent smaller than the
    polygon. Returns None when pygridgen is not installed.
    '''
    try:
        import pygridgen
    except ImportError:
        return None
    g = pygridgen.Gridgen(poly['x'].to_numpy(), poly['y'].to_numpy(),
                          poly['beta'].to_numpy(), shape=shape, precision=1e-5,
                          verbose=False)
    x, y = np.asarray(g.x), np.asarray(g.y)
    # split each cell into two triangles (as testvolume in grid_IC.py)
    a1 = 0.5 * np.abs((x[:-1, :-1] - x[1:, :-1]) * (y[:-1, 1:] - y[:-1, :-1])
                      - (x[:-1, :-1] - x[:-1, 1:]) * (y[1:, :-1] - y[:-1, :-1]))
    a2 = 0.5 * np.abs((x[:-1, 1:] - x[1:, :-1]) * (y[1:, 1:] - y[:-1, 1:])
                      - (x[:-1, 1:] - x[1:, 1:]) * (y[1:, :-1] - y[:-1, 1:]))
    return np.sum(a1 + a2)


def scale_to_area(build, target_area, scale0=1.0, area=polygon_area, n_iter=5):
    '''Find the width factor that gives target_area (area is ~linear in it).'''
    scale = scale0
    for _ in range(n_iter):
        poly = to_polygon(*build(scale)[:2])
        scale *= target_area / area(poly)
    return scale


def main(n_meanders=N_MEANDERS, sinuosity=SINUOSITY, omega=None, shape=(200, 7),
         alexander_sigma=ALEXANDER_SMOOTH_SIGMA):
    bank_a, bank_b = read_alexander()
    mid, width, s = centreline(bank_a, bank_b)
    length = s[-1]
    origin = bank_a[0]
    t0 = mid[1] - mid[0]
    side = np.sign(cross2(t0, bank_b[0] - bank_a[0]))

    # --- Alexander: squared ends, then light smoothing of each bank
    sq_a, sq_b = square_ends(bank_a, bank_b)
    alex_a = smooth_bank(sq_a, alexander_sigma)
    alex_b = smooth_bank(sq_b, alexander_sigma)
    alexander = to_polygon(alex_a, alex_b)
    raw_area = polygon_area(to_polygon(sq_a, sq_b))
    shift = max(distance_to_polyline(alex_a, sq_a).max(), distance_to_polyline(alex_b, sq_b).max())
    print(f'Alexander bank smoothing: sigma {alexander_sigma:.0f} m, max bank shift {shift:.1f} m, '
          f'area {polygon_area(alexander):.0f} m2 vs {raw_area:.0f} m2 digitised '
          f'({100 * (polygon_area(alexander) / raw_area - 1):+.2f}%)')
    plot_alexander_smoothing(sq_a, sq_b, alex_a, alex_b, alexander_sigma,
                             './domains/alexander_smoothing.png')
    target = polygon_area(alexander)

    # --- NoMeander: straight bank A heading along the overall valley direction
    nm_dir = np.array([1.0, 0.0])        # eastwards, as in the original NoMeander
    nm_build = lambda k: build_nomeander(origin, nm_dir, side, s, width, k)
    nm_scale = scale_to_area(nm_build, target)
    # refine on the area of the generated grid (needs pygridgen), so the
    # three model grids, not just the polygons, have the same area
    target_grid = grid_area(alexander, shape)
    match_on = 'polygon'
    if target_grid is not None:
        match_on = 'grid'
        g_area = lambda poly: grid_area(poly, shape)
        nm_scale = scale_to_area(nm_build, target_grid, nm_scale, g_area, n_iter=2)
    nomeander = to_polygon(*nm_build(nm_scale))

    # --- Meander
    wavelength = length / n_meanders
    if omega is None:
        omega = omega_for_sinuosity(sinuosity)
    me_build = lambda k: build_meander(origin, nm_dir, side, s, width, k, wavelength, omega)
    me_scale = scale_to_area(me_build, target)
    if target_grid is not None:
        me_scale = scale_to_area(me_build, target_grid, me_scale, g_area, n_iter=2)
    me_a, me_b, r_min, half_w_max = me_build(me_scale)
    meander = to_polygon(me_a, me_b)

    # --- checks and outputs
    rows = []
    for name, poly, k in (('Alexander', alexander, 1.0),
                          ('Meander', meander, me_scale),
                          ('NoMeander', nomeander, nm_scale)):
        crossing = self_intersects(poly)
        if crossing:
            raise RuntimeError(f'{name} polygon self-intersects')
        assert poly['beta'].sum() == 4
        g = grid_area(poly, shape)
        rows.append({'domain': name, 'n_vertices': len(poly),
                     'area_m2': polygon_area(poly),
                     'area_vs_alexander_%': 100 * (polygon_area(poly) / target - 1),
                     'grid_area_m2': g if g is not None else np.nan,
                     'grid_area_vs_alexander_%': 100 * (g / target_grid - 1) if g is not None else np.nan,
                     'width_scale': k, 'centreline_m': length})
        poly.to_csv(f'./domains/{name}.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv('./domains/domain_summary.csv', index=False)
    print(f'Widths scaled to match the Alexander {match_on} area')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nMeander: wavelength {wavelength:.1f} m, max angle {np.degrees(omega):.1f} deg '
          f'(sinuosity {1 / bessel_j0(omega):.2f}); min radius of curvature '
          f'{r_min:.1f} m vs max half width {half_w_max:.1f} m')
    if r_min < half_w_max:
        print('WARNING: the inner bank folds over in the tightest bends')

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    for name, poly in (('Alexander', alexander), ('Meander', meander), ('NoMeander', nomeander)):
        p = pd.concat([poly, poly.iloc[:1]])
        axes[0].plot(p['x'], p['y'], lw=0.8, label=name)
        c = poly[poly['beta'] == 1]
        axes[0].plot(c['x'], c['y'], 'k.', ms=4)
    axes[0].plot(*origin, 'r*', ms=10, label='mouth')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title('Domains (ITM, m)')
    axes[1].plot(s, width, 'k-o', ms=3, label='Alexander paired width (unscaled)')
    axes[1].plot(s, nm_scale * width, '-', label=f'NoMeander width (x{nm_scale:.3f})')
    axes[1].plot(s, me_scale * width, '--', label=f'Meander width (x{me_scale:.3f})')
    axes[1].set_xlabel('Distance from mouth along centreline (m)')
    axes[1].set_ylabel('Width (m)')
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig('./domains/domains_comparison.png', dpi=150)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    main()
