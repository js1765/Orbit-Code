# config.py  ---------------------------------------------------------------
"""
Global constants / singletons that many scripts import.
Only *define* things here—do not compute or fetch anything heavy.
"""
import platform
import pathlib
import pydylan
import numpy as np





# ---- CR3BP system --------------------------------------------------------
earth = pydylan.Body("Earth")
moon  = pydylan.Body("Moon")
cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

R_earth = earth.radius / cr3bp.DU
R_moon  = moon.radius  / cr3bp.DU
mu1     = 1 - cr3bp.mu
mu2     = cr3bp.mu

# ----- amended potential on y = 0 plane -----------------------------------
def U_tilde(x: np.ndarray, y: float = 0.0,
            m1: float = mu1, m2: float = mu2) -> np.ndarray:
    return (-0.5*(x**2 + y**2)
            - m1/np.sqrt((x + m2)**2 + y**2)
            - m2/np.sqrt((x - m1)**2 + y**2)
            - 0.5*m1*m2)

# ----- v_x = 0 lunar‑crash branches ---------------------------------------
def crash_vy_branches(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.abs(x - (1 - mu2))
    denom   = r**2 - R_moon**2
    radic   = 2*mu2*R_moon*(1 - R_moon/r) / denom
    valid   = radic >= 0
    vy_plus =  np.where(valid,  np.sqrt(radic) - x + 1, np.nan)
    vy_minus=  np.where(valid, -np.sqrt(radic) - x + 1, np.nan)
    return vy_plus, vy_minus, valid

L1_info = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
L2_info= cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
E0 = 1/2 * (L2_info[0][3]**2 + L2_info[0][4]**2) + U_tilde(L2_info[0][0],L2_info[0][1],mu1,mu2)
jacobimin = -2*E0


# For demonstration, pick an energy E0 between L1 and L2 energies
E0_JPL = L2_info[1]
jacobimin_JPL = -2*L2_info[1]   # Example upper limit for Jacobi constant

earth_x = -mu2
moon_x = 1 - mu2
earth_collision_radius_km = 7000.0
earth_collision_radius = earth_collision_radius_km / cr3bp.DU
moon_collision_radius  = pydylan.Body("Moon").radius  / cr3bp.DU

# ---- file system paths ---------------------------------------------------

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent
# print(base_dir)

BASE_PATH = {
    # "Darwin": pathlib.Path("/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database"),
    # "Linux":  pathlib.Path("/home/jg3607/SSA_project/Orbit_database")
    # "Darwin": pathlib.Path("/Users/jackschwartz/Desktop/VSCode Files/Research Stuff/Jannik Code/Orbit_database-main")
    "Darwin": base_dir
}.get(platform.system())


if BASE_PATH is None:
    raise EnvironmentError("Unsupported OS")

DATA_DIR   = BASE_PATH / "data"
FIG_DIR    = BASE_PATH / "Figures" / "IC_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)   # auto‑create

