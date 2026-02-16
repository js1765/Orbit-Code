import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
import requests
import concurrent.futures
import pydylan
import pickle

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

# For demonstration, let's pick an energy E0 between L1 and L2 energies
E0 = 1.1 * L2_info[1]
#E0 = L1_info[1] 
# In many standard CR3BP notations:
mu1 = 1 - cr3bp.mu  # The 'Earth' fraction if cr3bp.mu ~ mass fraction of Moon
mu2 = cr3bp.mu      # The 'Moon' fraction

# ------------------------------------------------------------------------------
# Define the "amended potential" and distance functions
# ------------------------------------------------------------------------------
def r1(x, y, mu2):
    return np.sqrt((x + mu2)**2 + y**2)

def r2(x, y, mu1):
    return np.sqrt((x - mu1)**2 + y**2)

def U_tilde(x, y, mu1, mu2):
    R1 = r1(x, y, mu2)
    R2 = r2(x, y, mu1)
    return -0.5*(x**2 + y**2) - mu1/R1 - mu2/R2 - 0.5*mu1*mu2

# ------------------------------------------------------------------------------
# 1) Function to plot the accessible region in (x,y,vx) space, excluding Earth/Moon
# ------------------------------------------------------------------------------
def plot_accessible_region(ax, cr3bp, E0, nx=25, ny=25, nvx=25, alpha_cloud=0.015):
    mu = cr3bp.mu
    mu1 = 1 - mu
    mu2 = mu
    R_earth = earth.radius  / cr3bp.DU
    R_moon  = moon.radius / cr3bp.DU
    x_vals  = np.linspace(-0.9, 1.2, nx)
    y_vals  = np.linspace(-0.85, 0.85, ny)
    vx_vals = np.linspace(-6, 6, nvx)
    X, Y, VX = np.meshgrid(x_vals, y_vals, vx_vals, indexing='ij')
    Uvals       = U_tilde(X, Y, mu1, mu2)
    inside_sqrt = 2.0 * (E0 - Uvals) - VX**2
    accessible_mask = (inside_sqrt >= 0)
    earth_mask = (r1(X, Y, mu2) < R_earth)
    moon_mask  = (r2(X, Y, mu1) < R_moon)
    not_in_bodies_mask = ~(earth_mask | moon_mask)
    final_mask = accessible_mask & not_in_bodies_mask
    x_acc  = X[final_mask]
    y_acc  = Y[final_mask]
    vx_acc = VX[final_mask]
    ax.scatter(
        x_acc, y_acc, vx_acc,
        s=1,
        c='gray',
        alpha=alpha_cloud
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('v_x')
    ax.set_title(f'Accessible region and orbits in (x,y,vx) for E = {E0:.3f}')

# ------------------------------------------------------------------------------
# 2) Function to plot one or more periodic orbits in the same axes
# ------------------------------------------------------------------------------
def plot_periodic_orbits(ax, orbits, color='red', lw=1.5, label=None):
    """
    Plots one or more periodic orbits on the given 3D Axes object 'ax'.
    If a label is provided, only the first orbit is plotted with that label.
    Each orbit is assumed to be a 2D array with columns = (x, y, z, vx, vy, vz),
    where x is column 0, y is column 1, and vx is column 3.
    """
    if isinstance(orbits, np.ndarray) and orbits.ndim == 2:
        orbits = [orbits]
    for i, orbit in enumerate(orbits):
        x  = orbit[:, 0]
        y  = orbit[:, 1]
        vx = orbit[:, 3]
        lab = label if (i == 0 and label is not None) else None
        ax.plot(x, y, vx, color=color, linewidth=lw, label=lab)

# ------------------------------------------------------------------------------
# Duplicate get_phase_options and get_orbit for orbit propagation
# ------------------------------------------------------------------------------
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
    return results_orbit

# ------------------------------------------------------------------------------
# 3) Main plotting section
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Plot the accessible region
    plot_accessible_region(ax, cr3bp, E0, nx=50, ny=50, nvx=50, alpha_cloud=0.1)

    # 2) Plot a Lyapunov orbit for E0 (if in the valid energy range)
    if 2.741514474 < -2*E0 < 3.188341115:
        lyapunov = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, E0)
        assert lyapunov.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success 
        lyap_orbit = get_orbit(lyapunov.orbit_initial_state, lyapunov.orbit_period).states
        plot_periodic_orbits(ax, [lyap_orbit], color='red', lw=1, label='Lyapunov')

    # 3) Load and plot other periodic orbits from the JPL database via URL
    configs = [
        {
            'family': 'dro',
            'branch': None,
            'jacob_off': 0.001,
            'color': 'blue',
            'label': 'DRO'
        },
        {
            'family': 'dpo',
            'branch': None,
            'jacob_off': 0.001,
            'color': 'orange',
            'label': 'DPO'
        },
        {
            'family': 'resonant',
            'branch': 21,
            'jacob_off': 0.001,
            'color': 'black',
            'label': 'Resonant (2:1)'
        },
        {
            'family': 'resonant',
            'branch': 31,
            'jacob_off': 0.001,
            'color': 'green',
            'label': 'Resonant (3:1)'
        },
        {
            'family': 'resonant',
            'branch': 41,
            'jacob_off': 0.001,
            'color': 'brown',
            'label': 'Resonant (4:1)'
        },
    ]

    for cfg in configs:
        base_url = (
            f"https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
            f"?sys=earth-moon&family={cfg['family']}"
            f"&jacobimin={-2*E0 - cfg['jacob_off']:.4f}"
            f"&jacobimax={-2*E0 + cfg['jacob_off']:.4f}"
        )
        if cfg['branch'] is not None:
            base_url += f"&branch={cfg['branch']}"
        data = requests.get(base_url).json()
        if not data or 'data' not in data or not data['data']:
            print(f"Warning: No data returned for URL '{base_url}'. Skipping...")
            continue
        data_array = np.array(data['data'], dtype=float)
        energies = np.array([float(row[6]) for row in data['data']])
        best_index = np.argmin(np.abs(energies + 2*E0))
        initial_state = data_array[best_index][:6]
        orbit_period = data_array[best_index][7]
        orbit_states = get_orbit(initial_state, orbit_period).states
        plot_periodic_orbits(ax, [orbit_states], color=cfg['color'], lw=1, label=cfg['label'])

    # 4) Load resonant 5:1 and 6:1 orbits from pickle files and plot at most one each.
    target_energy = E0
    pickle_path_51 = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_5_1_database.pkl"
    pickle_path_61 = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_6_1_database.pkl"
    
    try:
        with open(pickle_path_51, "rb") as f:
            resonant_51_orbits = pickle.load(f)
        print(f"Loaded {len(resonant_51_orbits)} resonant 5:1 orbits from pickle.")
    except Exception as e:
        print(f"Error loading resonant 5:1 orbits: {e}")
        resonant_51_orbits = []
    
    try:
        with open(pickle_path_61, "rb") as f:
            resonant_61_orbits = pickle.load(f)
        print(f"Loaded {len(resonant_61_orbits)} resonant 6:1 orbits from pickle.")
    except Exception as e:
        print(f"Error loading resonant 6:1 orbits: {e}")
        resonant_61_orbits = []
    
    # For each, check if target_energy falls within the stored energy range.
    # Energy range for 5:1 reso orbits: [-2.041,-1.600]
    # Energy range for 6:1 reso orbits: [-2.194,-1.815]
    if resonant_51_orbits:
        energies_51 = np.array([orbit["jacobi_energy"] for orbit in resonant_51_orbits if "jacobi_energy" in orbit])
        if energies_51.size == 0:
            print("No jacobi_energy values in resonant 5:1 orbits; skipping...")
        elif target_energy < energies_51.min() or target_energy > energies_51.max():
            print("E0 is outside the energy range for resonant 5:1 orbits; skipping...")
        else:
            best_index = np.argmin(np.abs(energies_51 - target_energy))
            best_orbit = resonant_51_orbits[best_index]
            plot_periodic_orbits(ax, [best_orbit["states"]], color='magenta', lw=1, label='Resonant (5:1)')
    
    if resonant_61_orbits:
        energies_61 = np.array([orbit["jacobi_energy"] for orbit in resonant_61_orbits if "jacobi_energy" in orbit])
        if energies_61.size == 0:
            print("No jacobi_energy values in resonant 6:1 orbits; skipping...")
        elif target_energy < energies_61.min() or target_energy > energies_61.max():
            print("E0 is outside the energy range for resonant 6:1 orbits; skipping...")
        else:
            best_index = np.argmin(np.abs(energies_61 - target_energy))
            best_orbit = resonant_61_orbits[best_index]
            plot_periodic_orbits(ax, [best_orbit["states"]], color='cyan', lw=1, label='Resonant (6:1)')

    # Add legend and finish up.
    ax.legend(loc='upper right')
    plt.tight_layout()
    filename = f"/Users/jannik/Documents/PhD_Princeton/Research/SSA/Figures/Fixed_E0_plots/Orbits_energy_{E0:.3f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.show()