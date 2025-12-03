import numpy as np


from scipy.special import comb

def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i

def bezier_curve(points, nTimes=100):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.
    points should be a list of lists, or list of tuples
    such as [ [1,1], 
              [2,3], 
              [4,5], ..[Xn, Yn] ]
    nTimes is the number of time steps, defaults to 1000
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def decode_design(x, bounds, geom):
    """
    Map normalized optimizer vector x (Control Points)
    to smooth chord and pitch distributions using Bezier curves.
    
    x structure:
    - First half: Chord Control Points (y-values only, x is assumed uniform)
    - Second half: Twist Control Points (y-values only)
    """
    Ns = len(geom["rStations"])
    r_stations = np.array(geom["rStations"])
    
    # Determine number of control points
    # We assume x is split 50/50 between chord and twist
    n_cp = len(x) // 2
    
    # 1. Chord Distribution
    cmin_frac = bounds["chordMin"]
    cmax_frac = bounds["chordMax"]
    D = geom["diameter"]
    
    # Control points for chord (normalized 0-1)
    cp_chord_norm = x[:n_cp]
    
    # Create Bezier curve for Chord
    # X-coords of CPs: evenly spaced from 0 to 1 (representing r_stations)
    # Y-coords of CPs: the optimizer values
    cp_x = np.linspace(0, 1, n_cp)
    points_chord = list(zip(cp_x, cp_chord_norm))
    
    # Evaluate Bezier at the specific r_stations
    # We need to map r_stations (e.g. 0.15 to 1.0) to t (0 to 1)
    r_min = r_stations[0]
    r_max = r_stations[-1]
    t_vals = (r_stations - r_min) / (r_max - r_min)
    
    # Calculate Bezier values manually for specific t_vals
    # (Re-using logic from bezier_curve but for specific t)
    nPoints = n_cp
    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t_vals) for i in range(0, nPoints)])
    chord_curve_norm = np.dot(cp_chord_norm, polynomial_array)
    
    # Scale to physical dimensions
    chords = (cmin_frac * D) + chord_curve_norm * (cmax_frac * D - cmin_frac * D)

    # 2. Twist Distribution
    pmin = bounds["pitchDegMin"]
    pmax = bounds["pitchDegMax"]
    
    # Control points for twist
    cp_twist_norm = x[n_cp:]
    points_twist = list(zip(cp_x, cp_twist_norm))
    
    # Evaluate Bezier for Twist
    twist_curve_norm = np.dot(cp_twist_norm, polynomial_array)
    
    # Scale to degrees
    betas_le = pmin + twist_curve_norm * (pmax - pmin)

    return chords, betas_le
