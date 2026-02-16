#!/usr/bin/env python3
"""
plot_resonant_orbits_extended.py

This script plots orbit trajectories in the x-y plane.
It can load orbits either from a local pickle file or from an URL (propagating them),
or both. In addition, you can choose which type of orbit to plot:
  - resonant (with an additional option for the resonance, e.g. 5:1 or 6:1),
  - dros (distant retrograde),
  - dpos (distant prograde), or
  - lyapunov (around L1).
plt.grid(True)
For resonant, dros, and dpos, the orbits are either fetched from an API (URL)
or loaded from a pickle file. In the URL case, only every nth orbit is propagated
(to save time). For Lyapunov orbits the orbits are generated on the fly for Jacobi
constants between [2.741514474, 3.188341115] using a spacing adapted to the step
parameter.

The plotted trajectories are colored according to the initial x value (x₀) using
a smooth colormap. The final plot is saved to a file whose name encodes the orbit
family, energy interval (or Jacobi constant) and the downsampling step.

Usage examples:
    python plot_resonant_orbits_extended.py --source pickle   --orbit_type resonant --resonance 6:1 --step 1
    python plot_resonant_orbits_extended.py --source url      --orbit_type dros     --step 2
    python plot_resonant_orbits_extended.py --source both     --orbit_type lyapunov --step 3
"""

import argparse
import os
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import pydylan

# =============================================================================
# --- CR3BP and Orbit Propagation Functions ---
# =============================================================================

def get_cr3bp():
    """
    Set up and return the CR3BP environment using Earth and Moon.
    """
    earth = pydylan.Body('Earth')
    moon  = pydylan.Body('Moon')
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
    return cr3bp

def U_tilde(x: np.ndarray, y: float = 0.0,
            m1: float = 1 - get_cr3bp().mu, m2: float = get_cr3bp().mu) -> np.ndarray:
    return (-0.5*(x**2 + y**2)
            - m1/np.sqrt((x + m2)**2 + y**2)
            - m2/np.sqrt((x - m1)**2 + y**2)
            - 0.5*m1*m2)

def get_phase_options():
    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 20
    phase_options.match_point_position_constraint_tolerance = 1E-4
    phase_options.match_point_velocity_constraint_tolerance = 1E-5
    phase_options.match_point_mass_constraint_tolerance = 1E-3
    phase_options.control_coordinate_transcription = pydylan.enum.spherical
    return phase_options

def get_orbit(state_on_orbit, orbit_period, cr3bp):
    """
    Propagate an orbit given its initial state and orbital period.
    Returns a NumPy array with columns: [time, x, y, z, vx, vy, vz]
    """
    #state_on_orbit[3] = -state_on_orbit[3]
    #state_on_orbit[4] = -state_on_orbit[4]
    thruster_parameters = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300, Isp=1000, thrust=1)
    phase_options = get_phase_options()
    # Dummy control array; its structure must match pydylan expectations.
    orbit_period += 0.01
    zero_control = np.array([0, orbit_period, 0] + [0, 0, 0]*20 + [700])
    mission_start = pydylan.FixedBoundaryCondition(state_on_orbit)
    orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)
    orbit.add_phase_options(phase_options)
    orbit.set_thruster_parameters(thruster_parameters)
    results_orbit = orbit.evaluate_and_return_solution(zero_control)
    return results_orbit.states

# =============================================================================
# --- Functions to Fetch Orbits from URL ---
# =============================================================================

def fetch_orbits(orbit_type, jacobimin, resonance=None, lpo_branch = None, system="earth-moon"):
    """
    Fetch orbits from the API.
    
    Parameters:
        orbit_type (str): One of "resonant", "dros", "dpos", or "lyapunov".
        jacobimin (float): The lower bound for the Jacobi constant.
        resonance (str): For resonant orbits only (e.g. "5:1" or "6:1").
        system (str): The CR3BP system (default "earth-moon").
    
    Returns:
        NumPy array of orbit data. Each row is expected to have at least 8 columns:
          [x, y, z, vx, vy, vz, ..., T]
    """
    base_url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
    params = {"sys": system, "jacobimin": jacobimin}
    
    if orbit_type == "resonant":
        if resonance is None:
            raise ValueError("For resonant orbits, please specify --resonance (e.g. '5:1' or '6:1').")
        p_val, q_val = resonance.split(":")
        params["family"] = "resonant"
        # For the API, use a branch parameter (e.g. "51" for 5:1)
        params["branch"] = p_val + q_val
    elif orbit_type == "lyapunov":
        params["family"] = "lyapunov"
        params["libr"] = 1
    elif orbit_type == "dros":
        params["family"] = "dro"
    elif orbit_type == "dpos":
        params["family"] = "dpo"
    elif orbit_type == "lpo":
        params["family"] = "lpo"
        params["branch"] = lpo_branch
        #params["jacobimax"] = 3.5
    else:
        raise ValueError("Invalid orbit type specified.")

    response = requests.get(base_url, params=params)
    data_json = response.json()
    if "data" in data_json and len(data_json["data"]) > 0:
        data_array = np.array(data_json["data"], dtype=float)
        return data_array
    else:
        return np.array([]).reshape(0, 8)

# =============================================================================
# --- Functions to Load Orbits from a Pickle File ---
# =============================================================================

def load_pickle_orbits(orbit_type, resonance=None, retro=None):
    """
    Load orbits from the appropriate pickle file based on the orbit type.
    
    For resonant orbits, expects files like:
      - "resonant_orbits_5_1_database.pkl" for 5:1,
      - "resonant_orbits_6_1_database.pkl" for 6:1.
    For dros, dpos, or lyapunov orbits, expects files:
      - "dros_orbits_database.pkl",
      - "dpos_orbits_database.pkl",
      - "lyapunov_orbits_database.pkl",
    respectively.
    
    Each orbit is assumed to be a dictionary with keys "initial_state" and "states".
    """
    if orbit_type == "resonant":
        if resonance is None:
            raise ValueError("For resonant orbits, please specify --resonance (e.g. '5.0:1.0' or '6.0:1.0').")
        res_filename = resonance.replace(":", "_")
        pickle_file = f"//Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/reso_orbits_{res_filename}_retro_{retro}.pkl"
    elif orbit_type == "circular_moon":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/circular_orbits_moon_database.pkl"
    else:
        raise ValueError("Invalid orbit type specified for pickle loading.")

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file '{pickle_file}' not found.")

    with open(pickle_file, "rb") as f:
        orbits = pickle.load(f)
    print(f"Loaded {len(orbits)} orbits from '{pickle_file}'.")
    return orbits

# =============================================================================
# --- Plotting Function ---
# =============================================================================

def plot_orbits(orbit_trajectories):
    """
    Plot orbit trajectories in the x-y plane. Each orbit is colored
    according to its initial x value (x₀) using a continuous colormap.
    
    Assumes the input list is already downsampled if desired.
    """
    # No additional downsampling here.
    orbits_to_plot = orbit_trajectories

    # Gather initial x values for color mapping.
    energy_list = []   
    for orbit in orbits_to_plot:
        if "initial_state" in orbit:
            init_state = orbit["initial_state"]
        else:
            init_state = orbit["states"][0]
        energy_list.append(-(init_state[3]**2 + init_state[4]**2) - 2 * U_tilde(init_state[0]))

    energy_array = np.array(energy_list)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=energy_array.min(), vmax=energy_array.max())
    
    plt.figure(figsize=(10, 8))
    for orbit in orbits_to_plot:
        if "initial_state" in orbit:
            init_state = orbit["initial_state"]
        else:
            init_state = orbit["states"][0]
        color = cmap(norm(-(init_state[3]**2 + init_state[4]**2) - 2 * U_tilde(init_state[0])))
        # Plot columns: [time, x, y, z, vx, vy, vz] --> x vs y.
        x = orbit["states"][:, 0]
        y = orbit["states"][:, 1]
        plt.plot(x, y, lw=1.5, color=color)

    plt.plot(1-get_cr3bp().mu, 0, marker='o', markersize=10, markeredgewidth=3,  color='grey', linestyle='none') 
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(energy_array)           # tells the colorbar the data range

    # add the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label("C", fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("$r_1$ [DU]", fontsize=22)
    plt.ylabel("$r_2$ [DU]", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=20)
    #plt.legend(fontsize=16, markerscale=2)
    plt.grid(True)
    plt.tight_layout()

# =============================================================================
# --- Main Function ---
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot orbit trajectories in the x-y plane from pickle or URL sources."
    )
    parser.add_argument(
        "--orbit_type",
        type=str,
        choices=["resonant", "dros", "dpos", "lyapunov","circular_moon","lpo"],
        default="lyapunov",
        help="Type of orbit to plot: resonant, dros, dpos, or lyapunov. Default: resonant"
    )
    parser.add_argument(
        "--lpo_branch",
        type=str,
        choices=["E", "W"],
        default="E",
        help="Type of orbit to plot: resonant, dros, dpos, or lyapunov. Default: resonant"
    )
    parser.add_argument(
        "--resonance",
        type=str,
        default="8.0:1.0",
        help="For resonant orbits only: specify the resonance (e.g. '5:1' or '6:1'). Default: 6:1"
    )
    parser.add_argument(
        "--retro",
        type=str,
        default="True",
        help="For resonant orbits only: specify if retrograde. Default: False"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["pickle", "url", "both"],
        default="url",
        help="Source of orbits: load from pickle file, from URL, or both. Default: url"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="Downsample: propagate (or load) every nth orbit (e.g. --step 2 means every 2nd orbit)."
    )
    parser.add_argument(
        "--jacobimin",
        type=float,
        default=3.17218,
        help="Jacobi constant lower bound for the API query (for non-lyapunov orbits)."
    )
    args = parser.parse_args()
    
    orbit_trajectories = []
    
    # Special handling for Lyapunov orbits: always generate them on the fly.
    if args.orbit_type == "lyapunov":
        cr3bp = get_cr3bp()
        # Define the energy (Jacobi constant) interval.
        E_low = -1/2*3.188 + 0.01
        E_high = -1/2* args.jacobimin
        # Adapt the number of samples based on the step parameter.
        num_orbits = max(2, int(200 / args.step))
        energies = np.linspace(E_low, E_high, num_orbits)
        print(f"Generating {len(energies)} Lyapunov orbits between E = {E_low} and {E_high}.")
        for energy in energies:
            lyapunov = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, energy)
            if lyapunov.solve_for_orbit() != pydylan.enum.OrbitGenerationResult.Success:
                print(f"Lyapunov orbit generation failed for energy {energy}.")
                continue
            trajectory = get_orbit(lyapunov.orbit_initial_state, lyapunov.orbit_period, cr3bp)
            orbit_trajectories.append({
                "initial_state": lyapunov.orbit_initial_state,
                "states": trajectory
            })
    else:
        # For other orbit types, use URL and/or pickle sources.
        # URL source: fetch and propagate only every nth orbit.
        if args.source in ("url", "both"):
            cr3bp = get_cr3bp()
            url_data = fetch_orbits(args.orbit_type, jacobimin=args.jacobimin,
                                    resonance=args.resonance if args.orbit_type == "resonant" else None,
                                    lpo_branch=args.lpo_branch if args.orbit_type=="lpo" else None)
            print(f"Fetched {len(url_data)} orbits from URL for {args.orbit_type} orbits.")
            if args.step > 1:
                url_data = url_data[::args.step]
                print(f"Propagating every {args.step}th orbit, resulting in {len(url_data)} orbits.")
            for row in url_data:
                if len(row) < 8:
                    continue
                ic = np.array(row[:6])
                T = row[7]
                try:
                    trajectory = get_orbit(ic, T, cr3bp)
                except Exception as e:
                    print(f"Orbit propagation failed: {e}")
                    continue
                orbit_trajectories.append({
                    "initial_state": ic,
                    "states": trajectory
                })
        # Pickle source: load orbits (they are assumed already downsampled if desired).
        if args.source in ("pickle", "both"):
            try:
                pickle_orbits = load_pickle_orbits(args.orbit_type,
                                                   resonance=args.resonance if args.orbit_type=="resonant" else None,
                                                   retro=args.retro if args.orbit_type=="resonant" else None)
                pickle_orbits = pickle_orbits[::args.step]
            except Exception as e:
                print(f"Error loading pickle orbits: {e}")
                pickle_orbits = []
            orbit_trajectories.extend(pickle_orbits)
    
    if len(orbit_trajectories) == 0:
        print("No orbits loaded. Exiting.")
        return


    # Plot the orbits (no further downsampling occurs here).
    plot_orbits(orbit_trajectories)
    
    # Construct a filename for saving the plot.
    if args.orbit_type == "resonant":
        family_str = f"resonant_{args.resonance.replace(':', '_')}_retro_{args.retro}"
        extra_info_str = f"jacobimin{args.jacobimin:.3f}"
    elif args.orbit_type == "lyapunov":
        family_str = "lyapunov"
        extra_info_str = f"E{E_low:.3f}-{E_high:.3f}"
    else:
        family_str = args.orbit_type
        extra_info_str = f"jacobimin{args.jacobimin:.3f}"
    
    filename = f"/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/Figures/periodic_orbit_families/{family_str}_step{args.step}_{extra_info_str}.pdf"
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.show()

if __name__ == "__main__":
    main()