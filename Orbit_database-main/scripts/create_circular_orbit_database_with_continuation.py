import numpy as np
import matplotlib.pyplot as plt
import pickle
import pydylan


def get_cr3bp():
    """
    Set up and return the CR3BP environment using Earth and Moon.
    """
    earth = pydylan.Body('Earth')
    moon  = pydylan.Body('Moon')
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
    return cr3bp

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
    #orbit_period += 0.01
    zero_control = np.array([orbit_period, 0, 0] + [0, 0, 0]*20 + [700])
    mission_start = pydylan.FixedBoundaryCondition(state_on_orbit)
    orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)
    orbit.add_phase_options(phase_options)
    orbit.set_thruster_parameters(thruster_parameters)
    results_orbit = orbit.evaluate_and_return_solution(zero_control)
    return results_orbit.states

# -------------------------------------------------------------------
# 2) Single-shooting "targeter" function
#    Given x0, guess vy0, guess T, integrate and return the constraints:
#        F1 = y(T)
#        F2 = vx(T)
#    We'll drive F1->0, F2->0 by adjusting vy0, T.
# -------------------------------------------------------------------
def shooting_constraints(vy0, T, x0):
    # Initial conditions
    state0 = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
    
    # Integrate from 0 to T
    sol = get_orbit(state0, T, cr3bp)
    
    xf, yf, zf, vxf, vyf, vzf = sol[-1,:6]
    
    # We want y(T)=0 and vx(T)=0
    F1 = yf      # y(T)
    F2 = vxf     # vx(T)
    
    return F1, F2, sol

# -------------------------------------------------------------------
# 3) Finite-difference Jacobian
#    We'll numerically approximate dF/d(vy0,T).
# -------------------------------------------------------------------
def finite_difference_jac(vy0, T, x0, eps=1e-6):
    # Base
    F1, F2, _ = shooting_constraints(vy0, T, x0)
    
    # dF/d(vy0)
    F1_vy0p, F2_vy0p, _ = shooting_constraints(vy0+eps, T, x0)
    dF1_dvy0 = (F1_vy0p - F1)/eps
    dF2_dvy0 = (F2_vy0p - F2)/eps
    
    # dF/dT
    F1_Tp, F2_Tp, _ = shooting_constraints(vy0, T+eps, x0)
    dF1_dT = (F1_Tp - F1)/eps
    dF2_dT = (F2_Tp - F2)/eps
    
    # Construct 2x2 Jacobian
    J = np.array([[dF1_dvy0, dF1_dT],
                  [dF2_dvy0, dF2_dT]])
    
    return J


# -------------------------------------------------------------------
# 4) Simple Newton iteration for the 2D system:
#       F1(vy0, T)=0
#       F2(vy0, T)=0
#    We'll do up to max_iter, stop if below a tolerance.
# -------------------------------------------------------------------
def differential_corrector(x0, vy0_guess, T_guess,
                           max_iter=20, tol=1e-10):
    vy0 = vy0_guess
    T   = T_guess
    
    for i in range(max_iter):
        # Evaluate constraints at current guess
        F, G, _ = shooting_constraints(vy0, T, x0)  # F1=F, F2=G
        res = np.array([F, G])
        norm_res = np.linalg.norm(res, ord=2)
        
        print(f"Iteration {i}: vy0={vy0:.6e}, T={T:.6e},  ||F||={norm_res:.3e}")
        
        if norm_res < tol:
            print("Converged!")
            break
        
        # Approximate Jacobian
        J = finite_difference_jac(vy0, T, x0, eps=1e-6)
        
        # Solve J * delta = -res
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            print("Jacobian is singular. Aborting.")
            break
        
        # Update unknowns
        vy0 += delta[0]
        T   += delta[1]
        
        # Safety check: T must stay positive
        if T <= 0:
            T = T_guess
            print(f"Warning: T became non-positive; resetting to {T}.")
    
    # Final evaluation
    F1, F2, sol = shooting_constraints(vy0, T, x0)
    final_res = np.linalg.norm([F1, F2])
    
    print(f"Final: vy0={vy0}, T={T}, residual={final_res}")
    return vy0, T, sol

def refine_half_period_guess(x0, vy0, T_guess):
    """
    1) Integrate from t=0 to t=2*T_guess (the naive guess for the full period).
    2) Find the *first* crossing of the x-axis (y=0) for t>0.
    3) Return that crossing time as an improved guess for the "half period."

    Assumes the initial state is [x0, 0, 0, vy0].
    """
    # Increase initial guess by a factor of 3 to ensure x-axis crossing
    T_guess = 3*T_guess
    # 1) Integrate
    state_init = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
    full_trajectory = get_orbit(state_init, T_guess, cr3bp)
    
    # full_trajectory: shape (N, 4) = [ [x(0),y(0),vx(0),vy(0)], ..., [x(2T), y(2T), ...] ]
    # We'll search for the time index i where y changes sign around 0 (excluding i=0).
    
    ys = full_trajectory[:,1]  # all y-values
    # We want the first index after 0 where y crosses from + to - or - to +.
    crossing_index = None
    
    for i in range(1, len(ys)):
        if ys[i-1]*ys[i] < 0:
            crossing_index = i - 1
            break
    
    if crossing_index is None:
        # No crossing found within [0, 2*T_guess] => fallback to T_guess
        return T_guess
    
    # 2) y changes sign between crossing_index and crossing_index+1
    # We'll do a simple linear interpolation to get exact crossing time fraction
    y0_prev = ys[crossing_index]
    y0_next = ys[crossing_index+1]
    frac = -y0_prev / (y0_next - y0_prev)  # fraction between t[i-1], t[i]
    
    # The time array is not directly stored in get_orbit's result,
    # but we can reconstruct from the known step. Let's see:
    # We used t_eval = np.linspace(0, 2*T_guess, n_points),
    # so each index i corresponds to t_i = i*(2*T_guess/(n_points-1)).
    N = len(full_trajectory)       # e.g. 800
    dt = T_guess/(N - 1)         # step in time
    crossing_time = (crossing_index + frac)*dt
    
    # crossing_time is the time after t=0 that y=0. This is presumably the "half" orbit.
    return crossing_time


cr3bp = get_cr3bp()
mu = cr3bp.mu
mu_earth = pydylan.Body("Earth").mu
mu1, mu2 = 1 - cr3bp.mu, cr3bp.mu

moon_plot = True
continuation = True
increasing = True

orbit_database = []

i = 0
vy0_corrected = 0

x0_values = np.arange(-mu - 0.4, -mu + 0.6, 0.01)
x0_values = x0_values[np.abs(x0_values) > 0.005]

safe_value = 1.09
lower_bound = 1.06
upper_bound = 1.131
step = 0.001

x0 = safe_value
vy0_corrected = 0.0
while x0 >= lower_bound:    # Distance from Earth (at x = 1 - mu) in the rotating frame
    print("x0: ", x0)
    if x0 < 0.8:
        # Distance from Earth (at x = 1 - mu) in the rotating frame
        r = abs(x0 + mu)
        circ_mu = mu1
        vy0 = np.sqrt(circ_mu / r) - (x0 + mu) 

    else:   
        r = abs(x0 - (1 - mu)) 
        circ_mu = mu2
        if continuation and i != 0:
            vy0 = vy0_corrected
        else:
            vy0 = np.sqrt(circ_mu / r) - (x0 - (1-mu))

    print("vy0 guess: ", vy0)

    T_2body =  2.0 * np.pi * np.sqrt(abs(x0 + mu)**3 / circ_mu) 
    print("Half-period guess:", T_2body/2)
    
    T_refined = refine_half_period_guess(x0, vy0, T_2body)
    print("Refined half-period guess:", T_refined)

    vy0_corrected, T_corrected, half_orbit = differential_corrector(x0,vy0,T_refined,max_iter=15,tol=1e-12)
    if i == 0:
        vy0_safe = vy0_corrected
    # Integrate forward
    state_init = np.array([x0, 0.0, 0.0, 0.0, vy0_corrected, 0.0])
    energy = 1/2 * (state_init[3]**2 + state_init[4]**2) + U_tilde(state_init[0],state_init[1],mu1,mu2)
    print("Orbital energy= ", energy)
    print("Below E2? ", energy <= cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)[1])
    orbit_data = get_orbit(state_init, T_corrected * 2, cr3bp)
    
    # Build a dictionary with the orbit information.
    orbit_dict = {
        "initial_state": state_init,
        "orbit_energy": energy,
        "orbital_period": T_corrected,
        "states": orbit_data
    }
    orbit_database.append(orbit_dict)

    if increasing:
        if x0 + step >= upper_bound:
            increasing = False
            x0 = safe_value - step
            vy0_corrected = vy0_safe
        else:
            x0 += step
    else:
        x0 -= step
    i += 1




#filename = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/data/circular_orbits_moon_database.pkl"

# from pathlib import Path

# base_dir = Path(__file__).resolve().parent.parent
# filename = base_dir / "data" / "circular_orbits_moon_database.pkl"

from support.constants import BASE_PATH
filename = BASE_PATH / "data" / "circular_orbits_moon_database.pkl"


# Save the entire database to a pickle file.
with open(filename, "wb") as f:
    pickle.dump(orbit_database, f)