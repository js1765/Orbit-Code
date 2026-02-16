"""
CR3BP Crash Map — 3D (x0, vx0, vy0)

Sets up the Earth–Moon CR3BP (PyDyLAN) and, for each (x0, vx0, vy0), sweeps a coarse set of 
initial velocity values to test collision outcomes. A triplet is labeled:
  • 'Crash Earth' if it crashes into Earth 
  • 'Crash Moon'  if it crashes into Moon 
  • 'Safe' otherwise 

Energy gate: if total energy > E2 (L2 threshold), the triplet is skipped.

Flags:
  • is_2d   : If True, fixes vx0 = 0.0 and performs a 2D sweep in (x0, vy0). 
              If False, sweeps a 3D grid over (x0, vx0, vy0). 
  • do_plot : If True, generates a 2D/3D scatter plot of crash regions. 
              If False, saves the crash initial conditions to pickle files instead.

Output:
  • 2D or 3D plot of crash regions (if do_plot=True) 
  • Pickle files of all crash points (if do_plot=False)
"""



import numpy as np
import pickle
from support.helpers import propagate_orbit
from support.constants import mu1, mu2, DATA_DIR, earth_x, moon_x, earth_collision_radius, moon_collision_radius, E0, U_tilde
from support.plot import plot_collision_scatter_2D, plot_collision_scatter_3D

def run_collision_check(is_2d, do_plot):
    x0_values = np.arange(-mu2 - 1.0, -mu2 + 1.3, 0.2)
    vy0_values = np.arange(-6.0, 6.0, 0.2)
    if is_2d:
        vx0_values = [0.0]
    else:
        vx0_values = np.arange(-6.0, 6.0, 2.0)

    # Lists to store results
    crash_earth_x, crash_earth_vy, crash_earth_vx = [], [], [] 
    crash_moon_x,  crash_moon_vy,  crash_moon_vx   = [], [], []
    safe_x,        safe_vy,        safe_vx        = [], [], []

    for x0 in x0_values:
        for vy0 in vy0_values:
            for vx0 in vx0_values:
                print("x0: ", x0)
                state_init = np.array([x0, 0.0, 0.0, vx0, vy0, 0.0])
                energy = 1/2 * (state_init[3]**2 + state_init[4]**2) + U_tilde(state_init[0],state_init[1],mu1,mu2)

                if energy <= E0:
                    if x0<0.75:
                        prop_time = np.abs(x0)*np.pi
                    else:
                        prop_time = np.abs(x0-(1-mu2))*np.pi
                    orbit_data = propagate_orbit(state_init, prop_time)
                    
                    # Distances from Earth and Moon
                    x_vals = orbit_data[:, 1]
                    y_vals = orbit_data[:, 2]

                    dist_from_earth = np.sqrt((x_vals - earth_x)**2 + y_vals**2)
                    dist_from_moon  = np.sqrt((x_vals - moon_x)**2  + y_vals**2)

                    earth_collision = np.any(dist_from_earth < earth_collision_radius)
                    moon_collision  = np.any(dist_from_moon  < moon_collision_radius)

                    if earth_collision:
                        crash_earth_x.append(x0)
                        crash_earth_vy.append(vy0)
                        crash_earth_vx.append(vx0)
                    elif moon_collision:
                        crash_moon_x.append(x0)
                        crash_moon_vy.append(vy0)
                        crash_moon_vx.append(vx0)
                    else:
                        safe_x.append(x0)
                        safe_vy.append(vy0)
                        safe_vx.append(vx0)

    if do_plot:
        if len(vx0_values) == 1:
            plot_collision_scatter_2D(crash_earth_x, crash_earth_vy, crash_moon_x, crash_moon_vy)
        else:
            plot_collision_scatter_3D(crash_earth_x, crash_earth_vx, crash_earth_vy, crash_moon_x,  crash_moon_vx,  crash_moon_vy)

    else:
         # -----------------------------------------------
        # Save Earth-crash and Moon-crash (x0, vy0) pairs to pickle
        # -----------------------------------------------
        if is_2d:
            earth_crash_data = list(zip(crash_earth_x, crash_earth_vy))
            moon_crash_data  = list(zip(crash_moon_x,  crash_moon_vy))
            with open(f"{DATA_DIR}/earth_crash_ICs_x_vy.pkl", "wb") as f_earth:
                pickle.dump(earth_crash_data, f_earth)#

            with open(f"{DATA_DIR}/moon_crash_ICs_x_vy.pkl", "wb") as f_moon:
                pickle.dump(moon_crash_data, f_moon)
        else:
            earth_crash_data = list(zip(crash_earth_x, crash_earth_vx, crash_earth_vy))
            moon_crash_data  = list(zip(crash_moon_x,  crash_moon_vx, crash_moon_vy))
            with open(f"{DATA_DIR}/earth_crash_ICs_x_vx_vy.pkl", "wb") as f_earth:
                pickle.dump(earth_crash_data, f_earth)

            with open(f"{DATA_DIR}/moon_crash_ICs_x_vx_vy.pkl", "wb") as f_moon:
                pickle.dump(moon_crash_data, f_moon)

        print("Saved Earth-crash initial conditions to 'earth_crash_ICs.pkl'.")
        print("Saved Moon-crash initial conditions to 'moon_crash_ICs.pkl'.")

run_collision_check(is_2d=True, do_plot=True)