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

################################################################################
#                   Timeout-Protected solve_for_orbit                          #
################################################################################
def _orbit_worker(x_val, p_val, q_val, retro, output_queue):
    """
    Worker function to be run in a separate process. It reconstructs 
    the CR3BP and ResonanceOptions objects internally, calls solve_for_orbit(),
    and puts a tuple (success, orbit_initial_state, orbit_energy, orbit_period)
    into output_queue.
    """
    try:
        # Reconstruct CR3BP
        cr3bp = get_cr3bp()
        
        # Create and configure your ResonanceOptions
        res_orbit_options = pydylan.periodic_orbit.ResonanceOptions()
        res_orbit_options.x = x_val
        res_orbit_options.p = p_val
        res_orbit_options.q = q_val
        res_orbit_options.retrograde = retro
        
        # Attempt the orbit solve
        reso_orbit = pydylan.periodic_orbit.Resonance(cr3bp, res_orbit_options)
        result = reso_orbit.solve_for_orbit()

        # Check result
        if result == pydylan.enum.OrbitGenerationResult.Success:
            # If success, gather orbit data
            orbit_initial_state = reso_orbit.orbit_initial_state
            orbit_energy = reso_orbit.orbit_energy
            orbit_period = reso_orbit.orbit_period
            states = reso_orbit.orbit_state_history
            stab_index = reso_orbit.get_stability_index()
            output_queue.put((True, orbit_initial_state, orbit_energy, orbit_period, states, stab_index))
        else:
            # If solver returned a non-success code
            output_queue.put((False, None, None, None, None, None))

    except Exception:
        # If an exception occurs, let the parent know it's a failure
        output_queue.put((False, None, None, None, None, None))

def solve_for_orbit_with_timeout(x_val, p_val, q_val, retro, time_limit=300):
    """
    Spawns a worker process that tries solve_for_orbit for up to time_limit seconds.
    If it times out or fails, this raises an Exception.
    Returns (orbit_initial_state, orbit_energy, orbit_period) upon success.
    """
    output_queue = mp.Queue()
    worker = mp.Process(
        target=_orbit_worker,
        args=(x_val, p_val, q_val, retro, output_queue)
    )
    worker.start()

    try:
        success, orbit_initial_state, orbit_energy, orbit_period, orbit_states, stab_index = \
            output_queue.get(timeout=time_limit)
    except mp.queues.Empty:
        # If no data arrives within time_limit, assume a timeout
        worker.terminate()
        worker.join()
        raise TimeoutError(f"solve_for_orbit() timed out after {time_limit} seconds for x={x_val}")

    # Now that we've drained the queue, we can safely join the child
    worker.join()
    if worker.exitcode != 0:
        # Child had a non-zero exit code (crash or forced termination)
        raise RuntimeError(f"Child process exited with code {worker.exitcode} for x={x_val}")

    if not success:
        raise RuntimeError(f"solve_for_orbit() failed or returned a non-success for x={x_val}.")

    return orbit_initial_state, orbit_energy, orbit_period, orbit_states, stab_index


################################################################################
#                              Main Script                                     #
################################################################################
if __name__ == "__main__":
     # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run CR3BP resonance orbit generation.")
    parser.add_argument("--p", type=float, default=8.0, help="Resonance parameter p.")
    parser.add_argument("--q", type=float, default=1.0, help="Resonance parameter q.")
    parser.add_argument("--retro", action="store_true", default=True, help="If given, specifies the resonant orbit is retrograde.")
    parser.add_argument("--start", type=float, default=0.01, help="Start value for x positions.")
    parser.add_argument("--stop", type=float, default=0.47, help="Stop value for x positions.")
    args = parser.parse_args()

    # Determine file path based on the operating system.
    system = platform.system()
    if system == "Darwin":  # macOS
        data_path = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data_new/"
    elif system == "Linux":  # Linux computing cluster
        data_path = "/home/jg3607/SSA_project/Orbit_database/data/"
    else:
        raise EnvironmentError("Unsupported operating system: only Darwin and Linux are supported.")
    
    filename = f"{data_path}reso_orbits_{args.p}_{args.q}_retro_{args.retro}.pkl"

    # Define the range of initial x positions
    start = args.start
    stop = args.stop
    step = 0.001
    x_starts = np.arange(start, stop + 0.001, step)
    # Example exclusion: x_starts = x_starts[(x_starts < 0.886) | (x_starts > 0.890)]
    
    orbit_database = []
    x_fail_list = []
    print(bool(args.retro))
    for reso_x in x_starts:
        # Try to solve for the orbit in a child process with a 5-min limit
        try:
            orbit_initial_state, orbit_energy, orbit_period, states, stab_index = solve_for_orbit_with_timeout(
                x_val=reso_x, 
                p_val=args.p, 
                q_val=args.q,
                retro=args.retro,
                time_limit=600  # 10 minutes
            )
        except TimeoutError:
            print(f"Orbit generation timed out for x = {reso_x}")
            x_fail_list.append(reso_x)
            continue
        except Exception as e:
            print(f"Orbit generation failed for x = {reso_x} with error: {e}")
            x_fail_list.append(reso_x)
            continue
        
        # Build and store a dictionary with orbit data
        orbit_dict = {            "initial_state": orbit_initial_state,
            "jacobi_energy": -2 * orbit_energy,
            "orbital_period": orbit_period,
            "states": states,
            "stability_index": stab_index,
        }
        orbit_database.append(orbit_dict)
        
        print("Initial state:", orbit_initial_state)
        print("Jacobi energy:", -2 * orbit_energy)
    
    # Save all successful orbits to a pickle
    with open(filename, "wb") as f:
        pickle.dump(orbit_database, f)
    
    print(f"Orbit database saved to {filename}")

    # Example of printing max/min energies
    if orbit_database:
        max_jacobi = max(orbit["jacobi_energy"] for orbit in orbit_database)
        min_jacobi = min(orbit["jacobi_energy"] for orbit in orbit_database)
        print(f"Max Jacobi energy = {max_jacobi}")
        print(f"Min Jacobi energy = {min_jacobi}")
    print(f"Orbits diverged or timed out for x0 = {x_fail_list}")