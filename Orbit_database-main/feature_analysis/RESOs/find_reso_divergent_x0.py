import multiprocessing as mp
import numpy as np
import time

import pydylan

TIMEOUT_SECONDS = 300  # 5 minutes

def get_cr3bp():
    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    return pydylan.eom.CR3BP(primary=earth, secondary=moon)

def _solve_for_orbit_worker(x_val, p, q, output_queue):
    """
    Worker function that recreates cr3bp and res_orbit_options from scratch,
    then calls solve_for_orbit() and puts the result in output_queue.
    """
    try:
        # Recreate your CR3BP
        cr3bp = get_cr3bp()

        # Recreate and configure your ResonanceOptions
        res_orbit_options = pydylan.periodic_orbit.ResonanceOptions()
        res_orbit_options.x = x_val
        res_orbit_options.p = p
        res_orbit_options.q = q
        res_orbit_options.retrograde = False

        # Create your Resonance object and attempt solve
        reso_orbit = pydylan.periodic_orbit.Resonance(cr3bp, res_orbit_options)
        result = reso_orbit.solve_for_orbit()

        # Put the result into the queue
        output_queue.put(result)
    except Exception:
        # If anything goes wrong, put a marker in the queue so the caller knows.
        output_queue.put(None)

def test_convergence(x_val, p, q):
    """
    Spawns a new process to attempt solve_for_orbit for up to TIMEOUT_SECONDS.
    We only pass in x_val, p, q, which are pickleable.
    """
    x_val = round(x_val, 3)  # Ensure 3 decimals

    output_queue = mp.Queue()
    p_worker = mp.Process(target=_solve_for_orbit_worker, args=(x_val, p, q, output_queue))
    p_worker.start()

    # Wait up to TIMEOUT_SECONDS
    p_worker.join(TIMEOUT_SECONDS)

    if p_worker.is_alive():
        # Timed out; kill the worker
        p_worker.terminate()
        p_worker.join()
        return False

    # Process ended in time; get the result
    result = output_queue.get()
    if result is None:
        return False

    return (result == pydylan.enum.OrbitGenerationResult.Success)

def find_threshold(x1, x2, p, q, tolerance=0.001):
    """
    Same logic as before, but calling the new test_convergence signature
    that doesn't take CR3BP or non-pickleable objects.
    """
    x1 = round(x1, 3)
    x2 = round(x2, 3)

    conv1 = test_convergence(x1, p, q)
    conv2 = test_convergence(x2, p, q)

    if conv1 == conv2:
        raise ValueError("One endpoint must converge, the other must not.")

    # Determine direction of the search
    if x1 < x2:
        if conv1:
            lower, upper = x1, x2
            convergent_is_lower = True
        else:
            lower, upper = x1, x2
            convergent_is_lower = False
    else:
        if conv1:
            lower, upper = x2, x1
            convergent_is_lower = False
        else:
            lower, upper = x2, x1
            convergent_is_lower = True

    # Binary search
    while round(upper - lower, 3) > tolerance:
        mid = round((lower + upper) / 2.0, 3)
        if mid == lower or mid == upper:
            break

        if test_convergence(mid, p, q):
            if convergent_is_lower:
                lower = mid
            else:
                upper = mid
        else:
            if convergent_is_lower:
                upper = mid
            else:
                lower = mid

    return round(lower if convergent_is_lower else upper, 3)

if __name__ == "__main__":
    # Example usage
    p_val = 3.0 
    q_val = 1.0
    x_val1 = 0.95
    x_val2 = 0.98

    threshold = find_threshold(x_val1, x_val2, p_val, q_val, tolerance=0.001)
    print("Threshold x for convergence:", threshold)