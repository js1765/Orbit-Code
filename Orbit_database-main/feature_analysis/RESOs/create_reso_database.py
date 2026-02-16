import pydylan
import numpy as np
import pickle
import multiprocessing as mp
import platform
import argparse

################################################################################
#                           Helper Functions                                   #
################################################################################

def get_cr3bp():    
    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    return pydylan.eom.CR3BP(primary=earth, secondary=moon)
    
def get_phase_options():
    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 20
    phase_options.match_point_position_constraint_tolerance = 1E-4
    phase_options.match_point_velocity_constraint_tolerance = 1E-5
    phase_options.match_point_mass_constraint_tolerance = 1E-3
    phase_options.control_coordinate_transcription = pydylan.enum.spherical
    return phase_options

def get_orbit(state_on_orbit,orbit_period):
    cr3bp = get_cr3bp()
    thruster_parameters = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300, Isp=1000, thrust=1)
    phase_options = get_phase_options()
    zero_control = np.array([0, orbit_period, 0] + [0, 0, 0] * 20 + [700])

    mission_start = pydylan.FixedBoundaryCondition(state_on_orbit) 

    orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)  
    orbit.add_phase_options(phase_options)
    orbit.set_thruster_parameters(thruster_parameters)

    results_orbit = orbit.evaluate_and_return_solution(zero_control)
    return results_orbit

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run CR3BP resonance orbit generation.")
    parser.add_argument("--p", type=float, default=2.0, help="Resonance parameter p.")
    parser.add_argument("--q", type=float, default=1.0, help="Resonance parameter q.")
    parser.add_argument("--retro", type=bool, default=False, help="Specifies if resonant orbit is retrograde")
    parser.add_argument("--start", type=float, default=0.032, help="Start value for x positions.")
    parser.add_argument("--stop", type=float, default=0.033, help="Stop value for x positions.")
    args = parser.parse_args()

    # Determine file path based on the operating system.
    system = platform.system()
    if system == "Darwin":  # macOS
        data_path = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/"
    elif system == "Linux":  # Linux computing cluster
        data_path = "/home/jg3607/SSA_project/Orbit_database/data/"
    else:
        raise EnvironmentError("Unsupported operating system: only Darwin and Linux are supported.")
    
    filename = f"{data_path}resonant_orbits_{args.p}_{args.q}_database.pkl"
    # Get the CR3BP model.
    cr3bp = get_cr3bp()
    
    # Define the range of initial x positions.
    start = args.start
    stop = args.stop
    step = 0.001
    x_starts = np.arange(start, stop, step)

    # This list will hold our orbit dictionaries.
    orbit_database = []
    x_fail_list = []

    # Set up resonance options for the orbit.
    res_orbit_options = pydylan.periodic_orbit.ResonanceOptions()
    res_orbit_options.p = args.p
    res_orbit_options.q = args.q
    res_orbit_options.retrograde = args.retro
    
    # Loop over each x value.
    for reso_x in x_starts:
        res_orbit_options.x = reso_x
        reso_orbit = pydylan.periodic_orbit.Resonance(cr3bp, res_orbit_options)
        
        # Try to solve for the orbit.
        try:
            result = reso_orbit.solve_for_orbit()
        except Exception:
            print(f"Orbit generation failed for x = {reso_x}")
            x_fail_list.append(reso_x)
            continue
        
        # Obtain the orbit trajectory.
        states = reso_orbit.orbit_state_history
        if np.shape(states) == (0,0):
            continue 
        
        # Build a dictionary with the orbit information.
        orbit_dict = {
            "initial_state": reso_x,
            "jacobi_energy": reso_orbit.orbit_energy,
            "orbital_period": reso_orbit.orbit_period,
            "states": states,
            "stability_index": reso_orbit.get_stability_index()
        }
        orbit_database.append(orbit_dict)
        
        # Optional: print information about the current orbit.
        print("Initial state:", reso_x)
        print("Jacobi energy:", reso_orbit.orbit_energy)
    
    # Save the entire database to a pickle file.
    #with open(filename, "wb") as f:
    #    pickle.dump(orbit_database, f)
    
    print(f"Orbit database saved to {filename}")
    if orbit_database:
        max_jacobi = -2 * orbit_database[0]["jacobi_energy"]
        min_jacobi = -2 * orbit_database[-1]["jacobi_energy"]
        print(f"Max Jacobi energy = {max_jacobi}")
        print(f"Min Jacobi energy = {min_jacobi}")
    print(f"Orbits diverged for x0 = {x_fail_list}")
