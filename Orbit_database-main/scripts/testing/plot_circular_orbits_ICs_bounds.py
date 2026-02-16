import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle  
import platform
import os 
import glob
import pydylan

# ------------------------------------------------------------------------------
# Example: Set up the CR3BP environment
# ------------------------------------------------------------------------------
earth = pydylan.Body('Earth')
moon = pydylan.Body('Moon')
cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

# Extract Earth and Moon radii in CR3BP dimensionless units
R_earth = earth.radius / cr3bp.DU
R_moon  = moon.radius  / cr3bp.DU

# Libration points info (L1, L2, etc.)
L1_info = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
L2_info = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)

# For demonstration, pick an energy E0 between L1 and L2 energies
E0_JPL = L2_info[1]

# In many standard CR3BP notations:
mu = cr3bp.mu
mu1 = 1 - cr3bp.mu  # 'Earth' fraction if cr3bp.mu ~ mass fraction of Moon
mu2 = cr3bp.mu      # 'Moon' fraction

# -----------------------------------------------------------------------------
# User-defined parameters
# -----------------------------------------------------------------------------
jacobimin_JPL = -2*L2_info[1]   # Example upper limit for Jacobi constant
system = "earth-moon"
plot_second_crossings = True 

oper_system = platform.system()
if oper_system == "Darwin":  # macOS
    base_path = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/"
elif oper_system == "Linux":  # Linux computing cluster
    base_path = "/home/jg3607/SSA_project/Orbit_database/"
else:
    raise EnvironmentError("Unsupported operating system: only Darwin and Linux are supported.")

# ------------------------------------------------------------------------------
# Define the amended potential and helper functions
# ------------------------------------------------------------------------------
def r1(x, y, mu2):
    return np.sqrt((x + mu2)**2 + y**2)

def r2(x, y, mu1):
    return np.sqrt((x - mu1)**2 + y**2)

def U_tilde(x, y, mu1, mu2):
    R1 = r1(x, y, mu2)
    R2 = r2(x, y, mu1)
    return -0.5*(x**2 + y**2) - mu1/R1 - mu2/R2 - 0.5*mu1*mu2

def get_phase_options():
    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 20
    phase_options.match_point_position_constraint_tolerance = 1E-4
    phase_options.match_point_velocity_constraint_tolerance = 1E-5
    phase_options.match_point_mass_constraint_tolerance = 1E-3
    phase_options.control_coordinate_transcription = pydylan.enum.spherical
    return phase_options

def get_orbit(state_on_orbit, orbit_period):
    thruster_parameters = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300, Isp=1000, thrust=1)
    phase_options = get_phase_options()
    zero_control = np.array([0, orbit_period, 0] + [0, 0, 0] * 20 + [700])
    mission_start = pydylan.FixedBoundaryCondition(state_on_orbit) 
    orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)  
    orbit.add_phase_options(phase_options)
    orbit.set_thruster_parameters(thruster_parameters)
    results_orbit = orbit.evaluate_and_return_solution(zero_control)
    return np.column_stack((results_orbit.time, results_orbit.states))

def get_resonant_label(filepath):
    filename = os.path.basename(filepath)  # e.g. "reso_orbits_4.0_1.0_retro_True.pkl"
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    x = int(float(parts[2]))
    y = int(float(parts[3]))
    if parts[-1] == "True":
        return f"Retrograde {x}:{y}"
    else:
        return f"Prograde {x}:{y}"

# ------------------------------------------------------------------------------
# Utility: Find all y=0 crossings in a trajectory
# ------------------------------------------------------------------------------
def find_all_y0_crossings(trajectory):
    """
    Given a trajectory array of shape (M, 7): [t, x, y, z, vx, vy, vz],
    return a list of crossing states (each is an array of 7 values).
    """
    t_vals = trajectory[:, 0]
    x_vals = trajectory[:, 1]
    y_vals = trajectory[:, 2]
    z_vals = trajectory[:, 3]
    vx_vals = trajectory[:, 4]
    vy_vals = trajectory[:, 5]
    vz_vals = trajectory[:, 6]
    crossing_states = []
    for i in range(len(y_vals) - 1):
        y0, y1 = y_vals[i], y_vals[i+1]
        t0, t1 = t_vals[i], t_vals[i+1]
        if y0 == 0.0 and i != 0:
            crossing_states.append(trajectory[i])
        elif y0 * y1 < 0.0:
            alpha = abs(y0) / abs(y1 - y0)
            t_cross = t0 + alpha * (t1 - t0)
            x_cross  = x_vals[i]  + alpha*(x_vals[i+1] - x_vals[i])
            z_cross  = z_vals[i]  + alpha*(z_vals[i+1] - z_vals[i])
            vx_cross = vx_vals[i] + alpha*(vx_vals[i+1] - vx_vals[i])
            vy_cross = vy_vals[i] + alpha*(vy_vals[i+1] - vy_vals[i])
            vz_cross = vz_vals[i] + alpha*(vz_vals[i+1] - vz_vals[i])
            crossing_state = np.array([t_cross, x_cross, 0.0, z_cross, vx_cross, vy_cross, vz_cross])
            crossing_states.append(crossing_state)
    return crossing_states

# ------------------------------------------------------------------------------
# Function to fetch orbits from the JPL CR3BP Periodic Orbit API
# ------------------------------------------------------------------------------
def fetch_orbits(family, jacobimin, branch=None):
    base_url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
    params = {
        "sys": system,
        "family": family,
        "jacobimin": jacobimin
    }
    if family == "resonant" and branch is not None:
        params["branch"] = branch
    if family == "lpo" and branch is not None:
        params["branch"] = branch
    if family == "lyapunov":
        params["libr"] = 1
    response = requests.get(base_url, params=params)
    data_json = response.json()
    if "data" in data_json and len(data_json["data"]) > 0:
        data_array = np.array(data_json["data"], dtype=float)
        return data_array
    else:
        return np.array([]).reshape(0, 6)

# ------------------------------------------------------------------------------
# 2D Plotting Function
# ------------------------------------------------------------------------------
def plot_cr3bp_orbits(all_orbits, all_crossings, L2_info, E0, cr3bp, R_earth, families_str, 
                      earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                      plot_second_crossings=True, jacobimin=None, which="vy"):
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker_idx = {}

    def select_velocity(vx_vals, vy_vals):
        return vy_vals if which == "vy" else vx_vals

    for (x_vals, vx_vals, vy_vals, label) in all_orbits:
        if label not in label_to_color:
            label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
            label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
        c = label_to_color[label]
        m = marker_cycle[label_to_marker_idx[label]]
        velocity_data = select_velocity(vx_vals, vy_vals)
        if x_vals.size > 0:
            ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=5)

    already_labeled = set()
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            c = label_to_color.get(label, 'black')
            cross_label = label
            lbl = cross_label if cross_label not in already_labeled else None
            if lbl:
                already_labeled.add(cross_label)
            velocity_data = select_velocity(vx_vals, vy_vals)
            ax.scatter(x_vals, velocity_data, c=[c], marker='*', s=5, alpha=0.5, label=None)

    # Plot accessible region (mostly relevant for vy)
    mu1 = 1 - cr3bp.mu
    mu2 = cr3bp.mu
    x_plus = np.linspace(-cr3bp.mu + 2*R_earth, L2_info[0][0], 100)
    y_dot_plus_plus  = np.sqrt(2*(E0 - U_tilde(x_plus, 0, mu1, mu2)))
    y_dot_plus_minus = -np.sqrt(2*(E0 - U_tilde(x_plus, 0, mu1, mu2)))
    x_minus = np.linspace(-cr3bp.mu - 2*R_earth, -0.8, 100)
    y_dot_minus_plus  = np.sqrt(2*(E0 - U_tilde(x_minus, 0, mu1, mu2)))
    y_dot_minus_minus = -np.sqrt(2*(E0 - U_tilde(x_minus, 0, mu1, mu2)))

    ax.plot(x_minus, y_dot_minus_plus,  color='black', label="Accessible region")
    ax.plot(x_minus, y_dot_minus_minus, color='black')
    ax.plot(x_plus,  y_dot_plus_plus,   color='black')
    ax.plot(x_plus,  y_dot_plus_minus,  color='black')

    ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
    ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')

    ax.set_xlim(-0.8, L2_info[0][0])
    ax.set_ylim(-6, 6)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
    title_str = "y=0 Crossings in Earth–Moon CR3BP"
    if jacobimin is not None:
        title_str += f" (Jacobi >= {jacobimin})"
    ax.set_title(title_str)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    which_str = which if which in ["vx", "vy"] else "vy"
    plt.savefig(
        f"{base_path}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

# ------------------------------------------------------------------------------
# 3D Plotting Function
# ------------------------------------------------------------------------------
def plot_cr3bp_orbits_3d(all_orbits, all_crossings, L2_info, E0, cr3bp,
                         earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                         jacobimin, plot_second_crossings=True):
    """
    Creates a 3D scatter plot of orbits with axes (x, v_y, v_x) along with the 
    accessible region boundary surface.
    """
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2, 1, 1))
    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker_idx = {}

     # Plot the accessible region boundary surface.
    N_x   = 80
    N_th  = 60
    x_min = -0.8
    x_max = L2_info[0][0]
    vy_min, vy_max = -6, 6
    vx_min, vx_max = -6, 6
    
    # Plot orbits
    for (x_vals, vx_vals, vy_vals, label) in all_orbits:
        if label not in label_to_color:
            label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
            label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
        c = label_to_color[label]
        m = marker_cycle[label_to_marker_idx[label]]
        if x_vals.size > 0:
            mask = (x_vals >= x_min) & (x_vals <= x_max) & (vy_vals >= vy_min) & (vy_vals <= vy_max) & (vx_vals >= vx_min) & (vx_vals <= vx_max)
            ax.scatter(x_vals[mask], vy_vals[mask], vx_vals[mask], c=[c], marker=m, label=label, s=2)

    # Plot y=0 crossing points with star markers
    already_labeled = set()
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            c = label_to_color.get(label, 'black')
            cross_label = label + " (2nd crossing)"
            lbl = cross_label if cross_label not in already_labeled else None
            if lbl is not None:
                already_labeled.add(cross_label)
            mask = (x_vals >= x_min) & (x_vals <= x_max) & (vy_vals >= vy_min) & (vy_vals <= vy_max) & (vx_vals >= vx_min) & (vx_vals <= vx_max)
            ax.scatter(x_vals[mask], vy_vals[mask], vx_vals[mask], c=[c], marker='*', s=2, alpha=1.0, label=None)

    x_arr  = np.linspace(x_min, x_max, N_x)
    theta  = np.linspace(0, 2*np.pi, N_th)
    X, TH = np.meshgrid(x_arr, theta)
    mu1 = 1 - cr3bp.mu
    mu2 = cr3bp.mu
    U_vals = U_tilde(X, 0.0, mu1, mu2)
    val    = 2.0*(E0 - U_vals)
    val[val < 0] = np.nan
    R = np.sqrt(val)
    R_clipped = np.where(R > 6, 6, R)
    VX = R_clipped * np.cos(TH)
    VY = R_clipped * np.sin(TH)
    ax.plot_surface(X, VY, VX, alpha=0.2, color='gray', edgecolor='none')

    ax.scatter(earth_crash_x0, earth_crash_vx0, earth_crash_vy0, s=5, alpha = 0.01, color='red',    label='Crash into Earth')
    ax.scatter(moon_crash_x0,  moon_crash_vx0, moon_crash_vy0,  s=5, alpha = 0.01, color='purple', label='Crash into Moon')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(vy_min, vy_max)
    ax.set_zlim(vx_min, vx_max)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$")
    ax.set_zlabel(r"$v_x$")
    ax.set_title(f"Earth–Moon CR3BP (Jacobi ≥ {jacobimin}) - 3D Plot")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(
        f"{base_path}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_vx_vy_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

E0 = 1/2 * (L2_info[0][3]**2 + L2_info[0][4]**2) + U_tilde(L2_info[0][0],L2_info[0][1],mu1,mu2)
jacobimin = -2*E0
# ------------------------------------------------------------------------------
# Collect orbits from the API for each family/branch
# ------------------------------------------------------------------------------
all_orbits = []    # List of tuples: (x_init, vx_init, vy_init, label)
all_crossings = [] # List of tuples for y=0 crossing points

# ------------------------------------------------------------------------------
# Load resonant orbits from pickle files
# ------------------------------------------------------------------------------
data_dir = f"{base_path}/data"
pattern = os.path.join(data_dir, "reso_orbits_*.pkl")
pickle_files = []
pickle_files.append(("Circular", f"{data_dir}/circular_orbits_-0.4_0.6_database.pkl"))

for label_res, pickle_file in pickle_files:
    try:
        with open(pickle_file, "rb") as f:
            pickle_orbits = pickle.load(f)
        print(f"Loaded {len(pickle_orbits)} orbits from '{pickle_file}'.")
    except Exception as e:
        print(f"Error loading pickle file '{pickle_file}': {e}")
        pickle_orbits = []
    
    if pickle_orbits:
        x_init_pickle, vx_init_pickle, vy_init_pickle, vy_min_pickle, vy_max_pickle = [], [], [], [], []
        x_cross_pickle, vx_cross_pickle, vy_cross_pickle = [], [], []
        for orbit in pickle_orbits:
            energy = -2*orbit["orbit_energy"] if label_res == "Circular" else orbit["jacobi_energy"]
            if energy >= jacobimin:
                ic = orbit["initial_state"]
                T  = orbit["orbital_period"]
                states = orbit["states"]
                x_init_pickle.append(ic[0])
                vx_init_pickle.append(ic[3])
                vy_init_pickle.append(ic[4])
                vy_min = np.sqrt(mu1*(2/(abs(ic[0]+mu))-2/(abs(ic[0]+mu)+abs(ic[0]+mu)-0.05))) - (ic[0]+mu)
                vy_min_pickle.append(vy_min)
                vy_max = np.sqrt(mu1*(2/(abs(ic[0]+mu))-2/(abs(ic[0]+mu)+abs(ic[0]+mu)+0.05))) - (ic[0]+mu)
                vy_max_pickle.append(vy_max)
                times = np.linspace(0, T, states.shape[0])
                crossings = find_all_y0_crossings(np.column_stack([times, states]))
                for state in crossings:
                    x_cross_pickle.append(state[1])
                    vx_cross_pickle.append(state[4])
                    vy_cross_pickle.append(state[5])
        all_orbits.append((np.array(x_init_pickle), np.array(vx_init_pickle), np.array(vy_init_pickle), label_res))
        all_orbits.append((np.array(x_init_pickle), np.array(vx_init_pickle), np.array(vy_min_pickle), "lower bound"))
        all_orbits.append((np.array(x_init_pickle), np.array(vx_init_pickle), np.array(vy_max_pickle), "upper bound"))
        if plot_second_crossings and len(x_cross_pickle) > 0:
            all_crossings.append((np.array(x_cross_pickle), np.array(vx_cross_pickle), np.array(vy_cross_pickle), label_res))


earth_crash_x0, earth_crash_vx0, earth_crash_vy0 = ([], [], [])
moon_crash_x0,  moon_crash_vx0, moon_crash_vy0  = ([], [], [])
families_str = "circ_bounds"

# ------------------------------------------------------------------------------
# Plotting calls
# ------------------------------------------------------------------------------
plot_cr3bp_orbits(all_orbits, all_crossings, L2_info, E0, cr3bp, R_earth, families_str, 
                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                  plot_second_crossings, jacobimin, "vy")
plot_cr3bp_orbits(all_orbits, all_crossings, L2_info, E0, cr3bp, R_earth, families_str, 
                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                  plot_second_crossings, jacobimin, "vx")
plot_cr3bp_orbits_3d(all_orbits, all_crossings, L2_info, E0, cr3bp, 
                    earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, jacobimin, plot_second_crossings)