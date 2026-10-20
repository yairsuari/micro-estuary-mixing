import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# pygridgen and geopandas are imported inside creategrid(), so the other
# functions (depth, get_H, get_S, ...) can be imported without them,
# e.g. by the ROMS input scripts in ROMS/python.
HERE = os.path.dirname(os.path.abspath(__file__))

DOMAINS = {
    # name: (domain polygon, POM initial-condition file)
    'Alexander': ('./domains/Alexander.csv', './IC_Alexander.dat'),
    'Meander': ('./domains/Meander.csv', './IC_Meander.dat'),
    'NoMeander': ('./domains/NoMeander.csv', './IC_No_Meander.dat'),
}

def depth(KB):
    """
    Establishes the vertical grid resolution based on a FORTRAN subroutine.

    This function creates a vertical grid with logarithmic distributions
    at the top and bottom and a linear distribution in between. It is a
    translation of a legacy FORTRAN subroutine.

   

    Args:
        kb (int): The number of vertical layers (KB in FORTRAN).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing four numpy arrays:
            - z: Vertical coordinates of layer interfaces (size kb).
            - zz: Vertical coordinates of layer centers (size kb).
            - dz: Thickness of each layer (size kb, last element is 0).
            - dzz: Distance between layer centers (size kb, last element is 0).
    """
    # Initialize arrays
    z = np.zeros(KB)
    zz = np.zeros(KB)
    dz = np.zeros(KB)
    dzz = np.zeros(KB)

    # Parameters from the original FORTRAN code
    kbm1 = KB - 1
    # NOTE: The original FORTRAN comment mentions KL1 = 0.3*KB as a default,
    # but the code hardcodes KL1 = 4. We follow the implementation.
    kl1 = 4
    kl2 = KB - 4

    if kl1 > kl2:
        raise ValueError(f"KB must be large enough so that KL1 ({kl1}) <= KL2 ({kl2}). KB must be at least {kl1 + 4}.")

    bb = float(kl2 - kl1) + 4.0
    cc = float(kl1) - 2.0
    ln2 = np.log(2)  # The FORTRAN code uses 0.693147, which is approx. ln(2)

    del1 = 2.0 / bb / np.exp(ln2 * float(kl1 - 2))
    del2 = 2.0 / bb / np.exp(ln2 * float(KB - kl2 - 1))

    # Set initial values (Z(1), ZZ(1) in FORTRAN)
    z[0] = 0.0
    zz[0] = -del1 / 2.0

    # Log distribution at the top (Corresponds to DO 3 K=2,KL1)
    for k in range(2, kl1 + 1):
        z[k - 1] = -del1 * np.exp(ln2 * float(k - 2))
        zz[k - 1] = -del1 * np.exp(ln2 * (float(k) - 1.50))

    # Linear distribution in the middle (Corresponds to DO 4 K=KL1,KL2)
    for k in range(kl1, kl2 + 1):
        z[k - 1] = -(float(k) - cc) / bb
        zz[k - 1] = -(float(k) - cc + 0.50) / bb

    # Log distribution at the bottom (Corresponds to DO 5 K=KL2,KBM1)
    for k in range(kl2, kbm1 + 1):
        z[k - 1] = (1.0 - del2 * np.exp(ln2 * float(KB - k - 1))) * -1.0
        zz[k - 1] = (1.0 - del2 * np.exp(ln2 * (float(KB - k) - 1.50))) * -1.0

    # Final adjustments from original code
    z[KB - 1] = -1.0                                  # Z(KB)
    zz[KB - 2] = -1.0 * (1.0 - del2 / 2.0)           # ZZ(KBM1) is overwritten
    zz[KB - 1] = -1.0 * (1.0 + del2 / 2.0)           # ZZ(KB)

    # Calculate layer thicknesses and distances between centers (vectorized)
    dz[:-1] = z[:-1] - z[1:]
    dzz[:-1] = zz[:-1] - zz[1:]

    return z, zz, dz, dzz

def write2IC(outvar,FileNames,varname):
    ''' 
    This function Writes the varibles calculated 
    in the code to text file'''
    print(varname)
    # loop filenames
    for filename in FileNames:
        # switch between number of        # dimensions in outvar
        if outvar.ndim == 1:
            # 1D array (e.g., z, zz, dz, dzz)
            with open(filename,mode='a') as f:
                f.write(f"{varname}\n")
                # Loop through the values of outvar and print each as a separate line to my file
                for k in range(outvar.shape[0]):
                    f.write(f"{outvar[k]:13.7E}\n")
                    
        elif outvar.ndim == 2:
            # 2D array (lon,lat,H,Wind stress)
            with open(filename,mode='a') as f:
                f.write(f"{varname}\n")
                # loop thouth the values of outvar and print each as seperate line to my file
                for j in range(outvar.shape[1]):
                    for i in range(outvar.shape[0]):
                        f.write(f"{outvar[i,j]:13.7E}\n")
        elif outvar.ndim == 3:
            # 3D array (T,S)
            with open(filename,mode='a') as f:
                f.write(f"{varname}\n")
                # Loop through the values of outvar in column-major (Fortran) order
                for k in range(outvar.shape[2]):
                    for j in range(outvar.shape[1]):
                        for i in range(outvar.shape[0]):
                            f.write(f"{outvar[i,j,k]:13.7E}\n")


def creategrid(domain,JM,IM):
    ''' 
    This function uses the pygridgen package to create 
    curvilinear orthogonal gridpoints os the domain'''
    import geopandas as gp
    import pygridgen
    x = domain['x'].to_numpy()
    y = domain['y'].to_numpy()
    beta = domain['beta'].to_numpy()
    grid = pygridgen.Gridgen(x, y, beta, shape=(IM, JM), precision=1e-5)
    # plt.plot(grid.x,grid.y,marker='.',linestyle='none',color='black')
    # plt.plot(x,y,marker='.',color='red')
    # plt.show()
# #    grid.x.shape (400, 7)
    # grid.x and grid.y are positions in EPSG:6984, convert them to WGS84 (EPSG:4326)
    # Convert to GeoDataFrame
    gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(grid.x.flatten(), grid.y.flatten()), crs="EPSG:6984")
    # Reproject to WGS84
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    # Extract lon and lat
    lon = gdf_wgs84.geometry.x.values.reshape((IM, JM))
    lat = gdf_wgs84.geometry.y.values.reshape((IM, JM))
    return lon, lat,grid.x,grid.y

def orient_grid(lon, lat, x, y, mouth_xy):
    '''
    pygridgen does not guarantee which end of the polygon becomes i=0.
    Flip the grid so that i=0 is at the mouth (the model's western/ocean
    boundary) and the (i,j) system is right-handed, i.e. j points 90 degrees
    counter-clockwise from i, as POM and ROMS expect. Flipping both axes is a
    180 degree rotation of the index space and keeps the cross-section
    symmetric profile unchanged.
    '''
    c = x.shape[1] // 2
    d_first = np.hypot(x[0, c] - mouth_xy[0], y[0, c] - mouth_xy[1])
    d_last = np.hypot(x[-1, c] - mouth_xy[0], y[-1, c] - mouth_xy[1])
    arrays = [lon, lat, x, y]
    if d_last < d_first:
        arrays = [a[::-1, :] for a in arrays]
    lon, lat, x, y = arrays
    xi = np.stack([x[1:, :-1] - x[:-1, :-1], y[1:, :-1] - y[:-1, :-1]])
    eta = np.stack([x[:-1, 1:] - x[:-1, :-1], y[:-1, 1:] - y[:-1, :-1]])
    handed = xi[0] * eta[1] - xi[1] * eta[0]
    if np.median(handed) < 0:
        lon, lat, x, y = [a[:, ::-1] for a in (lon, lat, x, y)]
        eta = np.stack([x[:-1, 1:] - x[:-1, :-1], y[:-1, 1:] - y[:-1, :-1]])
        xi = np.stack([x[1:, :-1] - x[:-1, :-1], y[1:, :-1] - y[:-1, :-1]])
        handed = xi[0] * eta[1] - xi[1] * eta[0]
    if np.any(handed <= 0):
        print(f'Warning: {np.sum(handed <= 0)} cells are not right-handed (folded grid)')
    return (np.ascontiguousarray(lon), np.ascontiguousarray(lat),
            np.ascontiguousarray(x), np.ascontiguousarray(y))

def check_and_smooth_H(H, max_r=0.2, max_passes=20):
    """
    Checks the bathymetry for steep slopes (PGE condition) and applies
    a simple 1-2-1 smoother until the condition is met. The 'r' parameter
    is defined as r = |H1 - H2| / (H1 + H2).

    Only the wet interior is checked and smoothed: the bank columns
    (j=0, j=JM-1) are the closed side walls and the first/last rows are
    the mouth bar and river end, which are set separately in get_H. Including
    the 0.1 m bank columns would force the whole cross-section towards 0.1 m.

    Args:
        H (np.ndarray): Bathymetry array (positive depth values), shape (IM, JM).
        max_r (float): Maximum allowed value for the slope parameter r.
        max_passes (int): Maximum number of smoothing passes.

    Returns:
        np.ndarray: Smoothed bathymetry.
    """
    H_smooth = np.copy(H)
    wet = H_smooth[1:-1, 1:-1]          # view on the interior wet cells
    print("\nSmoothing bathymetry (wet interior only) to reduce Pressure Gradient Error...")
    for pass_num in range(max_passes + 1):
        rx = np.abs(np.diff(wet, axis=0)) / (wet[1:, :] + wet[:-1, :])
        ry = np.abs(np.diff(wet, axis=1)) / (wet[:, 1:] + wet[:, :-1])
        max_rx = np.max(rx) if rx.size else 0.0
        max_ry = np.max(ry) if ry.size else 0.0
        print(f"Pass {pass_num}: max r along = {max_rx:.3f}, across = {max_ry:.3f}")
        if max(max_rx, max_ry) <= max_r:
            print(f"Success: Bathymetry meets the slope criterion (r <= {max_r}).")
            return H_smooth
        if pass_num == max_passes:
            break
        # 1-2-1 filter in the offending direction(s); the edges of the wet
        # block are mirrored so no dry/bank values leak in.
        if max_ry > max_r and wet.shape[1] > 1:
            p = np.pad(wet, ((0, 0), (1, 1)), mode='edge')
            wet[:] = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4.0
        if max_rx > max_r:
            p = np.pad(wet, ((1, 1), (0, 0)), mode='edge')
            wet[:] = (p[:-2, :] + 2 * p[1:-1, :] + p[2:, :]) / 4.0

    print(f"Warning: max r is still {max(max_rx, max_ry):.3f} after {max_passes} passes.")
    return H_smooth

def testvolume(x,y,H,domain):
    '''
    This function calculates the surface area, mean depth, and volume of the domain
    Which is represented as curvilinear orthogonal gridpoints
    
    INPUT:
    x - longitude in metric coordinate system
    y - latitude in metric coordinate system
    H - depth in each grid point x,y
    domain simply a string to identify the output between the three domains
    
    The calculation will be conducted by:
    1. Celecting each grid area (square) defined by four gridpoints
    2. Calculating the square surface area of each
    3. linearly interpolate H in the middle of the square from the values of four corners
    4. claculate the square volume by multiplying the interpolated H by the calculated area
    5. Sum are and volume of all of the squares in the domain
    6. print out the sum of area and volume, use domain as label
    '''
    total_area = 0.0
    total_volume = 0.0

    # Get coordinates and depths for the corners of all cells.
    # The shape of x, y, and H should be (IM, JM)
    x1 = x[:-1, :-1]
    y1 = y[:-1, :-1]
    x2 = x[:-1, 1:]
    y2 = y[:-1, 1:]
    x3 = x[1:, 1:]
    y3 = y[1:, 1:]
    x4 = x[1:, :-1]
    y4 = y[1:, :-1]

    h1 = H[:-1, :-1]
    h2 = H[:-1, 1:]
    h3 = H[1:, 1:]
    h4 = H[1:, :-1]

    # Split each quadrilateral into two triangles (p1,p2,p4 and p2,p3,p4)
    # and calculate the volume of the wedge under each.
    # Area of a triangle using shoelace formula: 0.5 * |(x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)|
    area1 = 0.5 * np.abs((x1 - x4) * (y2 - y1) - (x1 - x2) * (y4 - y1))
    volume1 = area1 * (h1 + h2 + h4) / 3.0

    area2 = 0.5 * np.abs((x2 - x4) * (y3 - y2) - (x2 - x3) * (y4 - y2))
    volume2 = area2 * (h2 + h3 + h4) / 3.0

    total_area = np.sum(area1 + area2)
    total_volume = np.sum(volume1 + volume2)

    mean_depth = total_volume / total_area if total_area > 0 else 0

    print(f"Domain: {domain}")
    print(f"Total Surface Area: {total_area:.2f} m^2")
    print(f"Total Volume: {total_volume:.2f} m^3")
    print(f"Mean Depth: {mean_depth:.2f} m")
    print()

def cross_section_factors(JM, outer_factor=0.6):
    '''
    Depth of each column as a fraction of the centreline depth. The wet
    columns 1..JM-2 decrease linearly from 1.0 at the centre to outer_factor
    next to the banks (JM=7 gives 0.6/0.8/1.0/0.8/0.6); the bank columns
    are set separately.
    '''
    c = JM // 2
    off = np.abs(np.arange(JM) - c)
    n_half = c - 1
    if n_half <= 0:
        return np.ones(JM)
    return 1.0 - (1.0 - outer_factor) * off / n_half


def get_H(x, y, IM, JM, l_lowres, z_lowres, plot_comparison=False, domain_name='',
          mouth_depth=0.01, river_end_depth=0.3, bank_depth=0.1, max_r=0.2):
    # H - bottome depth
    # This is a simplified method, I use the profile from my physics paper
    # (95th percentile of 70m sections) as the mid estuary depth and then 
    # create a simlified bottom (as number of index point from centerline increases depth decrease)
    if JM % 2 == 0 or JM < 3:
        raise ValueError('JM must be odd and >= 3 (symmetric cross-section around a centreline)')
    c = JM // 2

    # Calculate the cumulative distance along the centerline of the new grid (j=c)
    centerline_x = x[:, c]
    centerline_y = y[:, c]
    segment_lengths = np.sqrt(np.diff(centerline_x)**2 + np.diff(centerline_y)**2)
    L_H = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Interpolate depths from the low-resolution profile (l_lowres, z_lowres)
    # onto the new grid's centerline distances (L_H).
    centerline_depth = np.interp(L_H, l_lowres, z_lowres)

    H = centerline_depth[:, np.newaxis] * cross_section_factors(JM)[np.newaxis, :]
    H[:, 0] = bank_depth
    H[:, -1] = bank_depth

    # Store the original "ideal" bathymetry before smoothing
    H_original_positive = np.copy(H)

    # Smooth the bathymetry to meet the pressure gradient condition
    # The smoothing function expects positive depth values.
    H_smoothed_positive = check_and_smooth_H(H, max_r=max_r, max_passes=100)

    if plot_comparison:
        # Plot a comparison of a cross-section in the middle of the estuary
        plt.figure(figsize=(10, 6))
        mid_estuary_index = IM // 2
        plt.plot(H_original_positive[mid_estuary_index, :], 'r-o', label='Original Ideal Profile')
        plt.plot(H_smoothed_positive[mid_estuary_index, :], 'b-x', label='Smoothed Model Profile')
        plt.title(f'Bathymetry Profile Comparison for {domain_name} (mid-estuary)')
        plt.xlabel('J-index (cross-channel)')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        plt.savefig(f'bathy_smoothing_comparison_{domain_name}.png')
        plt.close()

    H = H_smoothed_positive
    # i=0 is the mouth (bar), i=IM-1 the river end
    H[0, 1:-1] = mouth_depth
    H[-1, 1:-1] = river_end_depth
# pom08 expects depth below msl as negative values    
    H=H*-1

    return H

def get_S(x, y, IM, JM, KB, H, z):
    # Initialize Temperature and Salinity arrays
    T = np.full((IM, JM, KB), 22.0)
    S = np.copy(T)

    # Create a 3D array of model grid point depths.
    # H is positive depth, z is negative (0 to -1). Depths will be negative.
    # z is either one vector of sigma fractions (POM) or an (IM, JM, K) array
    # of per-column fractions z/h (ROMS s-coordinates)
    z = np.asarray(z)
    Depths = H[:, :, np.newaxis] * (z if z.ndim == 3 else z[np.newaxis, np.newaxis, :])

    # Load and prepare the vertical salinity profile data
    df_TS = pd.read_excel(os.path.join(HERE, 'domains', 'TS_A4.xlsx'))
    # Sort by depth to ensure it's monotonically increasing for interpolation
    df_TS = df_TS.sort_values(by='Depth(m)').reset_index(drop=True)

    # Extract the west (ocean) and east (river) boundary salinity profiles
    S_west = df_TS['S'].to_numpy()
    s_depth = df_TS['Depth(m)'].to_numpy()
    S_east = np.zeros_like(S_west)  # Freshwater at the river end

    # Calculate the cumulative distance along the estuary centerline (j=JM//2)
    centerline_x = x[:, JM // 2]
    centerline_y = y[:, JM // 2]
    segment_lengths = np.sqrt(np.diff(centerline_x)**2 + np.diff(centerline_y)**2)
    dist_along_centerline = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = dist_along_centerline[-1]
    percentage_along_centerline = dist_along_centerline / total_length if total_length > 0 else np.zeros(IM)

    # Loop through the grid to set salinity at each point
    for i in range(IM):
        # Linearly interpolate between the west (ocean) and east (river) salinity profiles
        # based on the percentage of distance along the estuary.
        p = percentage_along_centerline[i] # p=0 at west (ocean), p=1 at east (river)
        s_current_profile = S_west * (1 - p) + S_east * p

        for j in range(JM):
            # Interpolate the current vertical profile onto the model's grid depths.
            # Use absolute depths for interpolation.
            model_depths_at_point = np.abs(Depths[i, j, :])
            S[i, j, :] = np.interp(model_depths_at_point, s_depth, s_current_profile, left=s_current_profile[0], right=s_current_profile[-1])    

    return T, S


def get_WIND(x, y, IM, JM, wind_speed=1.0, wind_direction=0.0):
    """
    Project wind onto a curvilinear orthogonal grid.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X-coordinates of grid points (IM x JM)
    y : numpy.ndarray  
        Y-coordinates of grid points (IM x JM)
    IM : int
        Number of grid points in I-direction
    JM : int
        Number of grid points in J-direction
    wind_speed : float
        Wind speed magnitude (default: 1.0 m/s)
    wind_direction : float
        Wind direction in degrees (meteorological convention: 0° = from North, 90° = from East)
        Default: 0.0 (wind from North)
    
    Returns:
    --------
    uwnd : numpy.ndarray
        Wind component in local I-direction (grid zonal)
    vwnd : numpy.ndarray
        Wind component in local J-direction (grid meridional)
    """
    
    # Initialize output arrays
    uwnd = np.zeros((IM, JM))
    vwnd = np.zeros((IM, JM))
    
    # Convert meteorological wind direction to mathematical convention
    # Meteorological: degrees from North, clockwise
    # Mathematical: degrees from East, counterclockwise
    wind_dir_math = 90.0 - wind_direction
    
    # Convert to radians
    wind_dir_rad = np.radians(wind_dir_math)
    
    # Wind components in Cartesian coordinates (East-North)
    wind_u_cart = wind_speed * np.cos(wind_dir_rad)  # Eastward component
    wind_v_cart = wind_speed * np.sin(wind_dir_rad)  # Northward component
    
    # Calculate grid orientation at each point
    for i in range(IM):
        for j in range(JM):
            # Handle boundary conditions for grid angle calculation
            if i < IM - 1:
                # Use forward difference
                dx = x[i+1, j] - x[i, j]
                dy = y[i+1, j] - y[i, j]
            else:
                # Use backward difference at the boundary
                dx = x[i, j] - x[i-1, j]
                dy = y[i, j] - y[i-1, j]
            
            # Calculate local grid angle (angle of I-direction relative to East)
            grid_angle = np.arctan2(dy, dx)
            
            # Rotation matrix to transform from Cartesian to local grid coordinates
            cos_theta = np.cos(grid_angle)
            sin_theta = np.sin(grid_angle)
            
            # Project wind onto local grid directions
            # uwnd: component along I-direction (local zonal)
            # vwnd: component along J-direction (local meridional, perpendicular to I)
            uwnd[i, j] = wind_u_cart * cos_theta + wind_v_cart * sin_theta
            vwnd[i, j] = -wind_u_cart * sin_theta + wind_v_cart * cos_theta
    
    return uwnd, vwnd


def get_WIND_vectorized(x, y, IM, JM, wind_speed=1.0, wind_direction=0.0):
    """
    Vectorized version of wind projection for better performance.
    
    Same parameters and returns as get_WIND() but uses numpy vectorized operations.
    """
    
    # Convert meteorological wind direction to mathematical convention
    wind_dir_math = 90.0 - wind_direction
    wind_dir_rad = np.radians(wind_dir_math)
    
    # Wind components in Cartesian coordinates
    wind_u_cart = wind_speed * np.cos(wind_dir_rad)
    wind_v_cart = wind_speed * np.sin(wind_dir_rad)
    
    # Calculate dx and dy for grid orientation
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    
    # Forward differences for most points
    dx[:-1, :] = x[1:, :] - x[:-1, :]
    dy[:-1, :] = y[1:, :] - y[:-1, :]
    
    # Backward differences for last row
    dx[-1, :] = x[-1, :] - x[-2, :]
    dy[-1, :] = y[-1, :] - y[-2, :]
    
    # Calculate grid angles
    grid_angles = np.arctan2(dy, dx)
    
    # Vectorized rotation
    cos_theta = np.cos(grid_angles)
    sin_theta = np.sin(grid_angles)
    
    # Project wind components
    uwnd = wind_u_cart * cos_theta + wind_v_cart * sin_theta
    vwnd = -wind_u_cart * sin_theta + wind_v_cart * cos_theta
    
    return uwnd, vwnd



def demo_wind():
    '''Example usage and testing of get_WIND on a small curved grid.'''
    # Create a simple test case with a curved grid
    IM, JM = 5, 3
    
    # Create a simple curved grid for testing
    i_indices = np.arange(IM)
    j_indices = np.arange(JM)
    I, J = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Example: slightly curved grid
    x = I * 1000.0 + 100.0 * np.sin(I * 0.1) * J
    y = J * 1000.0 + 50.0 * I
    
    # Test with 1 m/s wind from North (meteorological convention)
    uwnd, vwnd = get_WIND(x, y, IM, JM, wind_speed=1.0, wind_direction=0.0)
    
    print("Grid coordinates (x):")
    print(x)
    print("\nGrid coordinates (y):")
    print(y)
    print(f"\nWind components for 1 m/s from North:")
    print("uwnd (I-direction):")
    print(uwnd)
    print("vwnd (J-direction):")  
    print(vwnd)
    
    # Verify magnitude is preserved
    wind_magnitude = np.sqrt(uwnd**2 + vwnd**2)
    print(f"\nWind magnitude (should be ~1.0):")
    print(wind_magnitude)


def centreline_bathymetry(sampledist=70):
    '''
    Read the measured bathymetry and reduce it to a low-resolution centreline
    profile (mean of the values above the 95th percentile in a moving window).
    Returns distance from the mouth (m) and depth (m, positive).
    '''
    df_bathy = pd.read_csv('./domains/Alexanderwith_l.csv')
    # in df_bathy['Depth (m)'] replace nan values with 0.7
    df_bathy['Depth (m)'] = df_bathy['Depth (m)'].fillna(0.7)

    l=df_bathy['l'].to_numpy()
    z_bathy=df_bathy['Depth (m)'].to_numpy()
    l[0]=0.2
    l[-1]=0.2

    l_lowres=np.array(range(np.min(l),np.max(l),sampledist))
    z_lowres=np.zeros(l_lowres.shape[0])
    for k in range(0,l_lowres.shape[0]):
                ht=l>=(l_lowres[k]-2*sampledist)
                lt=l<(l_lowres[k]+2*sampledist)
                z_section=z_bathy[(ht & lt)]
                if z_section.size > 0:
                    z_min=np.percentile(z_section,95)
                    z_lowres[k]=z_section[z_section>z_min].mean()
    # replace nan valuex in z_lowres by 0.7
    z_lowres = np.nan_to_num(z_lowres, nan=0.7)
    return l_lowres, z_lowres


def main(IM=200, JM=7, KB=8, mouth_depth=0.01, river_end_depth=0.3, max_r=0.2,
         domains=tuple(DOMAINS), grid_dir='./grids'):
    '''
    Generate the grids and write the POM initial-condition files
    (IC_*.dat; layout Z, ZZ, DZ, DZZ, ALON, ALAT, H, T, S, UWND, VWND)
    and a grids/<domain>_grid.npz per domain for the ROMS scripts.
    '''
    os.makedirs(grid_dir, exist_ok=True)
    files = [DOMAINS[d][1] for d in domains]
    for filename in files:
        if os.path.exists(filename):
            os.remove(filename)

    # 1D vectors
    z, zz, dz, dzz = depth(KB)
    write2IC(z, files, 'Z')
    write2IC(zz, files, 'ZZ')
    write2IC(dz, files, 'DZ')
    write2IC(dzz, files, 'DZZ')

    l_lowres, z_lowres = centreline_bathymetry()

    for name in domains:
        polygon_file, ic_file = DOMAINS[name]
        domain = pd.read_csv(polygon_file, encoding='utf-8-sig')
        # the first two polygon vertices are the mouth corners (see make_domains.py)
        mouth_xy = (domain['x'].iloc[:2].mean(), domain['y'].iloc[:2].mean())
        lon, lat, x, y = orient_grid(*creategrid(domain, JM, IM), mouth_xy)

        # 2D fields, lon, lat, depth; i=0 is the mouth (west/ocean), i=IM-1 the river end
        write2IC(lon, [ic_file], 'ALON')
        write2IC(lat, [ic_file], 'ALAT')
        H = get_H(x, y, IM, JM, l_lowres, z_lowres, plot_comparison=True, domain_name=name,
                  mouth_depth=mouth_depth, river_end_depth=river_end_depth, max_r=max_r)
        print(f"Max depth (abs) for {name}: {np.max(np.abs(H))}")
        write2IC(H, [ic_file], 'H')
        testvolume(x, y, H, name)

        T, S = get_S(x, y, IM, JM, KB, H, z)
        write2IC(T, [ic_file], 'T')
        write2IC(S, [ic_file], 'S')

        uwnd, vwnd = get_WIND(x, y, IM, JM)
        write2IC(uwnd, [ic_file], 'UWND')
        write2IC(vwnd, [ic_file], 'VWND')

        np.savez(os.path.join(grid_dir, f'{name}_grid.npz'),
                 x=x, y=y, lon=lon, lat=lat, h=-H, T=T, S=S,
                 z=z, zz=zz, dz=dz, dzz=dzz, mouth_depth=mouth_depth,
                 river_end_depth=river_end_depth, crs='EPSG:6984')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate model grids and POM initial conditions')
    parser.add_argument('--im', type=int, default=200, help='points along the channel')
    parser.add_argument('--jm', type=int, default=7, help='points across the channel (odd)')
    parser.add_argument('--kb', type=int, default=8, help='number of sigma levels (POM KB)')
    parser.add_argument('--mouth-depth', type=float, default=0.01)
    parser.add_argument('--river-end-depth', type=float, default=0.3)
    parser.add_argument('--max-r', type=float, default=0.2)
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument('--demo-wind', action='store_true', help='run the get_WIND example and exit')
    args = parser.parse_args()
    if args.demo_wind:
        demo_wind()
    else:
        main(IM=args.im, JM=args.jm, KB=args.kb, mouth_depth=args.mouth_depth,
             river_end_depth=args.river_end_depth, max_r=args.max_r, domains=args.domains)
