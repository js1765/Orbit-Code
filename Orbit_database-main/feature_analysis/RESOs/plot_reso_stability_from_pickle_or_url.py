#!/usr/bin/env python3
"""
plot_stability_vs_energy.py

This script plots the stability index over orbital energy (Jacobi constant)
for a family of orbits. It loads orbits either from a local pickle file or from a URL
(or both). For pickle files, the stability index is stored under "stability_index"
(and if an "energy" value is not present it is computed from the initial state);
for URL data, the stability index is assumed to be at index 9 and the orbital energy at index 8.

Usage examples:
    python plot_stability_vs_energy.py --source pickle --orbit_type resonant --resonance 6:1 --step 1
    python plot_stability_vs_energy.py --source url    --orbit_type dros     --step 2
    python plot_stability_vs_energy.py --source both   --orbit_type lyapunov --step 3
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
    thruster_parameters = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300, Isp=1000, thrust=1)
    phase_options = get_phase_options()
    # Slightly increase the period to avoid numerical issues.
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

def fetch_orbits(orbit_type, jacobimin, resonance=None, system="earth-moon"):
    """
    Fetch orbits from the API.
    
    Parameters:
        orbit_type (str): "resonant", "dros", "dpos", or "lyapunov".
        jacobimin (float): Lower bound for the Jacobi constant.
        resonance (str): For resonant orbits only (e.g. "5:1" or "6:1").
        system (str): CR3BP system (default "earth-moon").
    
    Returns:
        NumPy array of orbit data. Expected to have at least 10 columns:
          [x, y, z, vx, vy, vz, T, <energy>, <orbital energy>, <stability index>]
    """
    base_url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
    params = {"sys": system, "jacobimin": jacobimin}
    
    if orbit_type == "resonant":
        if resonance is None:
            raise ValueError("For resonant orbits, please specify --resonance (e.g. '5:1' or '6:1').")
        p_val, q_val = resonance.split(":")
        params["family"] = "resonant"
        params["branch"] = p_val + q_val  # API expects branch as concatenated values.
    elif orbit_type == "lyapunov":
        params["family"] = "lyapunov"
        params["libr"] = 1
    elif orbit_type == "dros":
        params["family"] = "dro"
    elif orbit_type == "dpos":
        params["family"] = "dpo"
    else:
        raise ValueError("Invalid orbit type specified.")
    
    response = requests.get(base_url, params=params)
    data_json = response.json()
    if "data" in data_json and len(data_json["data"]) > 0:
        # Expect that the row contains at least 10 entries.
        data_array = np.array(data_json["data"], dtype=float)
        return data_array
    else:
        return np.array([]).reshape(0, 10)

# =============================================================================
# --- Functions to Load Orbits from a Pickle File ---
# =============================================================================

def load_pickle_orbits(orbit_type, resonance=None, retro = None):
    """
    Load orbits from the corresponding pickle file.
    For resonant orbits, expects a filename like "resonant_orbits_5_1_database.pkl".
    For dros, dpos, or lyapunov, expects files:
      "dros_orbits_database.pkl", "dpos_orbits_database.pkl", or "lyapunov_orbits_database.pkl".
    
    Each orbit is assumed to be a dictionary with keys "initial_state", "states",
    and "stability_index". Optionally, an "energy" entry may be present.
    """
    if orbit_type == "resonant":
        if resonance is None:
            raise ValueError("For resonant orbits, please specify --resonance (e.g. '5:1' or '6:1').")
        res_filename = resonance.replace(":", "_")
        pickle_file = f"/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/reso_orbits_{res_filename}_retro_{retro}.pkl"
    elif orbit_type == "dros":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/dros_orbits_database.pkl"
    elif orbit_type == "dpos":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/dpos_orbits_database.pkl"
    elif orbit_type == "lyapunov":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/lyapunov_orbits_database.pkl"
    else:
        raise ValueError("Invalid orbit type specified for pickle loading.")
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file '{pickle_file}' not found.")
    
    with open(pickle_file, "rb") as f:
        orbits = pickle.load(f)
    print(f"Loaded {len(orbits)} orbits from '{pickle_file}'.")
    return orbits

# =============================================================================
# --- Utility: Compute Jacobi Constant ---
# =============================================================================

def compute_jacobi_constant(state, mu):
    """
    Compute the Jacobi constant for a given state in the CR3BP.
    state: array-like [x, y, z, vx, vy, vz]
    mu: mass parameter of the CR3BP.
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    Omega = (1 - mu)/r1 + mu/r2 + 0.5*(x**2 + y**2)
    C = 2 * Omega - (vx**2 + vy**2 + vz**2)
    return C

# =============================================================================
# --- Plotting Function ---
# =============================================================================

def plot_stability_vs_energy(orbit_data, mu, orbit_type, resonance = None, retro = None):
    """
    Create a scatter plot of stability index (y-axis) versus orbital energy (x-axis).
    For each orbit, if an "energy" field is not present, it is computed using the 
    Jacobi constant formula.
    """
    energies = []
    stability_indices = []
    for orbit in orbit_data:
        if "jacobi_energy" in orbit:
            if orbit["jacobi_energy"] < 0:
                 jacobi_energy = -2 * orbit["jacobi_energy"]
            else:
                jacobi_energy = orbit["jacobi_energy"] # The saved jacobi energy is actually the energy
        else:
            jacobi_energy = compute_jacobi_constant(orbit["initial_state"], mu)
        stab = orbit["stability_index"]
        energies.append(jacobi_energy)
        stability_indices.append(stab)
    
    energies = np.array(energies)
    stability_indices = np.array(stability_indices)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(energies, stability_indices)
    plt.xlabel("Orbital Energy (Jacobi Constant)", fontsize="xx-large")
    plt.ylabel("Stability Index", fontsize="xx-large")
    plt.yscale("log")
    title = orbit_type
    if resonance is not None:
        title += f" {resonance}"
    if retro is not None:
        title += " " + ("retrograde" if retro == "True" else "prograde")
    plt.title(title, fontsize="xx-large")
    plt.grid(True)
    plt.tight_layout()

# =============================================================================
# --- Main Function ---
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot stability index vs orbital energy from orbit data (pickle or URL)."
    )
    parser.add_argument("--orbit_type", type=str, choices=["resonant", "dros", "dpos", "lyapunov"],
                        default="dpos", help="Type of orbit: resonant, dros, dpos, or lyapunov.")
    parser.add_argument("--resonance", type=str, default="4:1",
                        help="For resonant orbits only: specify the resonance (e.g. '5.0:1.0' or '6.0:1.0').")
    parser.add_argument("--retro", type=str, default="True",
                        help="For resonant orbits only: specify if retograde.")
    parser.add_argument("--source", type=str, choices=["pickle", "url", "both"],
                        default="url", help="Source of orbits: pickle, URL, or both.")
    parser.add_argument("--step", type=int, default=1,
                        help="Downsample: use every nth orbit (e.g. --step 2 means every 2nd orbit).")
    parser.add_argument("--jacobimin", type=float, default=3.17218, #3.17218
                        help="Jacobi constant lower bound for the API query (for non-lyapunov orbits).")
    args = parser.parse_args()
    
    orbit_data = []
    cr3bp = get_cr3bp()
    
    if args.orbit_type == "lyapunov":
        # Generate Lyapunov orbits on the fly.
        E_low = -1/2 * 3.188 + 0.01
        E_high = -1/2 * args.jacobimin
        num_orbits = max(2, int(200 / args.step))
        energies_lin = np.linspace(E_low, E_high, num_orbits)
        print(f"Generating {len(energies_lin)} Lyapunov orbits between E = {E_low} and E = {E_high}.")
        for energy in energies_lin:
            lyapunov = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, energy)
            if lyapunov.solve_for_orbit() != pydylan.enum.OrbitGenerationResult.Success:
                print(f"Lyapunov orbit generation failed for energy {energy}.")
                continue
            trajectory = get_orbit(lyapunov.orbit_initial_state, lyapunov.orbit_period, cr3bp)
            # Here we assume the Lyapunov object provides a stability_index attribute.
            orbit_data.append({
                "initial_state": lyapunov.orbit_initial_state,
                "states": trajectory,
                "energy": energy,
                "stability_index": lyapunov.stability_index
            })
    else:
        # For non-Lyapunov orbits, load from URL and/or pickle.
        if args.source in ("url", "both"):
            url_data = fetch_orbits(args.orbit_type, jacobimin=args.jacobimin,
                                    resonance=args.resonance if args.orbit_type == "resonant" else None)
            print(f"Fetched {len(url_data)} orbits from URL for {args.orbit_type} orbits.")
            if args.step > 1:
                url_data = url_data[::args.step]
                print(f"Using every {args.step}th orbit, resulting in {len(url_data)} orbits.")
            for row in url_data:
                if len(row) < 9:
                    continue
                ic = np.array(row[:6])
                T = row[7]
                energy = row[6]
                stab_index = row[8]
                try:
                    trajectory = get_orbit(ic, T, cr3bp)
                except Exception as e:
                    print(f"Orbit propagation failed: {e}")
                    continue
                orbit_data.append({
                    "initial_state": ic,
                    "states": trajectory,
                    "jacobi_energy": energy,
                    "stability_index": stab_index
                })
        if args.source in ("pickle", "both"):
            try:
                pickle_orbits = load_pickle_orbits(args.orbit_type,
                                                   resonance=args.resonance if args.orbit_type=="resonant" else None,
                                                   retro=args.retro if args.orbit_type == "resonant" else None)
                pickle_orbits = pickle_orbits[::args.step]
            except Exception as e:
                print(f"Error loading pickle orbits: {e}")
                pickle_orbits = []
            for orbit in pickle_orbits:
                orbit_data.append(orbit)
    
    if len(orbit_data) == 0:
        print("No orbits loaded. Exiting.")
        return

    # Plot stability index versus orbital energy.
    plot_stability_vs_energy(orbit_data, cr3bp.mu, args.orbit_type, 
                            resonance=args.resonance if args.orbit_type=="resonant" else None,
                            retro=args.retro if args.orbit_type == "resonant" else None)
    
    # Construct filename for saving the plot.
    if args.orbit_type == "resonant":
        family_str = f"resonant_{args.resonance.replace(':', '_')}_retro_{args.retro}"
        extra_info_str = f"jacobimin{args.jacobimin:.3f}"
    elif args.orbit_type == "lyapunov":
        family_str = "lyapunov"
        extra_info_str = "E_range"
    else:
        family_str = args.orbit_type
        extra_info_str = f"jacobimin{args.jacobimin:.3f}"
    
    filename = f"/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/Figures/stability_plots/{family_str}_stability_step{args.step}_{extra_info_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    #plt.show()

if __name__ == "__main__":
    main()