"""
helpers.py — CR3BP utility functions for PyDyLAN workflows
==========================================================

Overview
--------
Lightweight helpers for setting up and analyzing Earth–Moon CR3BP problems using PyDyLAN.
Functions cover: phase-option templates, single-phase propagation, y=0 Poincaré
crossings, analytic crash-surface branches, safe x-grid generation around body radii,
KD-tree–based region classification, transfer counting, truncated Gaussian sampling
respecting the energy/crash envelopes, and simple Bayesian post-processing.

Conventions & Dependencies
--------------------------
• Units are nondimensionalized (Earth–Moon distance, synodic time, etc.) as in PyDyLAN.
• Coordinates/velocities are in the rotating frame unless noted.
• Requires: numpy, matplotlib, PyDyLAN, scipy.spatial.KDTree, and project constants
  from support.constants (e.g., cr3bp, mu1, mu2, R_moon, U_tilde, ...).

Function inventory (quick reference)
------------------------------------
• get_phase_options() -> phase_options_structure
    Build a standard phase-options template (segments, tolerances, spherical control).

• propagate_orbit(state, period) -> (N,7) ndarray
    Single-phase propagation with fixed start/end boundary; returns [t, x, y, z, vx, vy, vz].

• find_all_y0_crossings(traj) -> list[(7,)]
    Extract all y=0 plane crossings from a trajectory via linear interpolation.

• moon_crash_vy_branches(x) -> (vy_plus, vy_minus, valid)
    Analytic v_y branches for Moon-impact boundary on the y=0 section as a function of x.

• earth_crash_vy_branches(x) -> (vy_plus, vy_minus, valid)
    Analytic v_y branches for Earth-impact boundary on the y=0 section as a function of x.

• crash_surface_vy(x, vx, mu_body, R_body, x_body) -> (x_valid, vy_plus, vy_minus)
    Generalized crash-surface branches for a given body and fixed v_x on y=0 section.

• generate_x_ranges(x_min, x_max, x_body, R_body, points=800, gap=1e-5) -> 1-D ndarray
    Build an x-grid that explicitly excludes the body disk [x_body − R_body, x_body + R_body].

• transfer_counts(before_xy, after_xy, fam_labels, classif_data, x_scale=1.0) -> (F+1,F+1)
    Row-normalized percentage matrix of transfers between families (last index = Crash).

• classify_slice_region(all_orb, all_cross, *, E0, x_scale, vxtol=2.5, Nx=550, Ny=550,
                        x_lims=(0.80,1.15), vy_lims=(-2.5,2.5))
    Build an (x, v_y) grid mask and nearest-neighbor family classification with DRO/DPO overrides.
    Returns (X, VY, mask, colour_idx, fam_labels, lut).

• sample_truncated_gaussian_old(mean, cov, N, E0, exclude_crash=True) -> (N,2)
    Rejection-sample (x, v_y) from a Gaussian within the energy envelope, optionally excluding crash.

• sample_truncated_gaussian(mean, cov, N, E0, exclude_crash=True, direction="both") -> (N,2)
    Like the above, with optional prograde/retrograde filtering relative to crash branches.

• sample_region_percentages(samples, classif_data, x_scale=1.0) -> (F+1,)
    Classify samples via KD-tree to compute percentages per family (+ Crash).

• calculate_posterior(ratio_calc_prior, measurement_ratios) -> (F,)
    Simple Bayesian update over region ratios: posterior ∝ prior ⊙ measurement.
"""

# helpers.py  --------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import pydylan
from support.constants import cr3bp, mu1, mu2, R_moon, earth_collision_radius, U_tilde, crash_vy_branches
# from .constants import cr3bp, mu1, mu2, R_moon, earth_collision_radius, U_tilde, crash_vy_branches
from scipy.spatial import cKDTree as KDTree


# ------------------ phase‑options template --------------------------------
def get_phase_options() -> pydylan.phase_options_structure:
    po = pydylan.phase_options_structure()
    po.number_of_segments = 20
    po.match_point_position_constraint_tolerance = 1e-4
    po.match_point_velocity_constraint_tolerance = 1e-5
    po.match_point_mass_constraint_tolerance    = 1e-3
    po.control_coordinate_transcription         = pydylan.enum.spherical
    return po

# ------------------ single‑phase propagation ------------------------------
def propagate_orbit(state: np.ndarray, period: float) -> np.ndarray:
    """Return (N,7) array [t,x,y,z,vx,vy,vz]."""
    thr = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300,
                                   Isp=1000, thrust=1)
    ctrl = np.zeros(3 + 3*20 + 1)
    ctrl[1] = period
    ctrl[-1] = 700

    bc  = pydylan.FixedBoundaryCondition(state)
    ms  = pydylan.Mission(cr3bp, bc, bc, pydylan.enum.snopt)
    ms.add_phase_options(get_phase_options())
    ms.set_thruster_parameters(thr)
    res = ms.evaluate_and_return_solution(ctrl)
    return np.column_stack((res.time, res.states[:,:6]))

# ------------------ y = 0 plane crossings ---------------------------------
def find_all_y0_crossings(traj: np.ndarray) -> list[np.ndarray]:
    t,x,y,z,vx,vy,vz = traj.T[:7,:]
    out=[]
    for i in range(len(y)-1):
        if y[i]==0 and i: out.append(traj[i]); continue
        if y[i]*y[i+1] < 0:
            alpha = abs(y[i]) / abs(y[i+1]-y[i])
            out.append(np.array([
                t[i]+alpha*(t[i+1]-t[i]),
                x[i]+alpha*(x[i+1]-x[i]),
                0.0,
                z[i]+alpha*(z[i+1]-z[i]),
                vx[i]+alpha*(vx[i+1]-vx[i]),
                vy[i]+alpha*(vy[i+1]-vy[i]),
                vz[i]+alpha*(vz[i+1]-vz[i])
            ]))
    return out

def moon_crash_vy_branches(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.abs(x - (1 - mu2))
    denom   = r**2 - R_moon**2
    radic   = 2*mu2*R_moon*(1 - R_moon/r) / denom
    valid   = radic >= 0
    vy_plus =  np.where(valid,  np.sqrt(radic) - x + 1, np.nan)
    vy_minus=  np.where(valid, -np.sqrt(radic) - x + 1, np.nan)
    return vy_plus, vy_minus, valid

def earth_crash_vy_branches(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.abs(x + mu2)
    denom   = r**2 - earth_collision_radius**2
    radic   = 2*mu1*earth_collision_radius*(1 - earth_collision_radius/r) / denom
    valid   = radic >= 0
    vy_plus =  np.where(valid,  np.sqrt(radic) - x, np.nan)
    vy_minus=  np.where(valid, -np.sqrt(radic) - x, np.nan)
    return vy_plus, vy_minus, valid

def crash_surface_vy(x, vx, mu_body, R_body, x_body):
    """
    Compute + and - branches of v_y (rotating frame) on y = 0 Poincaré section.

    Parameters
    ----------
    x : ndarray
        1‑D array of x‑coordinates.
    vx : float
        Fixed v_x value (rotating frame).
    mu_body : float
        Gravitational parameter (nondim) of the body considered (Earth: 1‑mu, Moon: mu).
    R_body : float
        Physical radius of the body, nondimensionalised by the Earth‑Moon distance.
    x_body : float
        x‑coordinate of the body's centre (−mu for Earth, 1‑mu for Moon).

    Returns
    -------
    tuple(ndarray, ndarray, ndarray)
        (x_valid, vy_plus, vy_minus) where x_valid contains the subset of x that
        yields a non‑negative radicand.
    """
    r = np.abs(x - x_body)
    denom = r**2 - R_body**2
    # avoid zero division
    denom = np.where(np.abs(denom) < 1e-15, np.nan, denom)

    radicand = (R_body**2 * vx**2 + 2 * mu_body * R_body * (1 - R_body / r)) / denom
    valid = radicand >= 0.0

    x_valid = x[valid]
    sqrt_term = np.sqrt(radicand[valid])

    vy_plus =  sqrt_term - x_valid + (x_body + mu2)    # upward branch (dot y > 0)
    vy_minus = -sqrt_term - x_valid + (x_body + mu2)    # downward branch (dot y < 0)

    return x_valid, vy_plus, vy_minus

def generate_x_ranges(x_min, x_max, x_body, R_body, points=800, gap=1e-5):
    """
    Produce an x‑grid that *excludes* the interval [x_body-R_body, x_body+R_body].
    Returns a concatenated 1‑D array (might be empty if the excluded band
    consumes the whole requested range).
    """
    left_edge  = x_body - R_body - gap
    right_edge = x_body + R_body + gap

    segments = []
    if x_min < left_edge:
        segments.append(np.linspace(x_min, left_edge,  points))
    if right_edge < x_max:
        segments.append(np.linspace(right_edge, x_max, points))

    return np.concatenate(segments) if segments else np.array([])

#def second_y0_crossing(x: float, vy: float,
#                       orbit_period: float = 2*np.pi) -> tuple[float,float] | None:
#    """
#    Propagate one slice point (x, vy) with v_x = 0 on the y=0 plane,
#    return the next y=0 crossing that has the *same sign* of vy.
#    If no such crossing occurs within 'orbit_period', return None.
#    """
#    state0 = np.array([x, 0.0, 0.0, 0.0, vy, 0.0])      # 6‑vector
#    traj   = propagate_orbit(state0, orbit_period)
#    cross  = find_all_y0_crossings(traj)
#
#    if not cross:
#        return None
#
#    sign0  = np.sign(vy)
#    for s in cross[1:]:                 # skip the t=0 crossing
#        if np.sign(s[5]) == sign0:
#            return s[1], s[5]           # (x, vy)
#    return None


#@overload
#def propagate_selected_points(classif_data, *, region:str,
#                              sample:int|None = None,
#                              period:float = 2*np.pi) -> tuple[np.ndarray, np.ndarray]: ...
#@overload
#def propagate_selected_points(classif_data, *, region:np.ndarray,
#                              sample:int|None = None,
#                              period:float = 2*np.pi) -> tuple[np.ndarray, np.ndarray]: ...


def transfer_counts(before_xy: np.ndarray,
                after_xy: np.ndarray,
                fam_labels: list[str],
                classif_data: tuple[np.ndarray, ...],
                x_scale: float = 1.0) -> np.ndarray:
    """
    Compute a percentage transfer matrix  (#before,fam_i → fam_j or Crash)
    classif_data  : (X, VY, mask, grid, fam_labels_per_idx, lut)
                    = output of classify_slice_region(...)
    x_scale       : scaling factor applied to the x‐distance in KD-tree queries
    Returns
    -------
    T  : (F+1, F+1) array; last index = crash region
    """
    X, VY, mask, grid, fam_order, _ = classif_data
    assert fam_order == fam_labels, "family label order mismatch"

    # --- build KD‐tree of the GRID using the same x_scale as classify_slice_region
    Xflat = X.ravel()
    VYflat = VY.ravel()
    Xscaled = (Xflat - mu1) * x_scale
    tree = KDTree(np.column_stack((Xscaled, VYflat)))
    idx_grid = grid.ravel()

    def classify_batch(pts: np.ndarray) -> np.ndarray:
        crash_label = len(fam_labels)
        finite_mask = np.isfinite(pts).all(axis=1)
        idx = np.full(pts.shape[0], crash_label, dtype=int)

        if finite_mask.any():
            good = pts[finite_mask]
            xg = (good[:, 0] - mu1) * x_scale
            vyg = good[:, 1]
            query = np.column_stack((xg, vyg))
            _, nn = tree.query(query, k=1)
            idx_valid = idx_grid[nn]
            idx_valid[idx_valid < 0] = crash_label
            idx[finite_mask] = idx_valid

        return idx

    idx_before = classify_batch(before_xy)
    idx_after  = classify_batch(after_xy)

    F = len(fam_labels) + 1
    T = np.zeros((F, F), dtype=float)
    for b, a, ay in zip(idx_before, idx_after, after_xy):
        if np.isnan(ay[0]):
            a = F - 1
        T[b, a] += 1.0

    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, 100 * T / row_sums, 0)
    return T

def classify_slice_region(all_orb, all_cross, *,
                          E0: float,
                          x_scale: float,
                          vxtol: float = 2.5,
                          Nx: int = 550,
                          Ny: int = 550,
                          x_lims: tuple[float, float] = (0.80, 1.15),
                          vy_lims: tuple[float, float] = (-2.5, 2.5)):
    """
    Return
    -------
    grid_x, grid_vy           : (Nx, Ny) meshgrids
    mask_inside               : boolean  (Nx, Ny)  valid cells
    colour_index              : int      (Nx, Ny)  family index for each cell
    fam_labels (ordered list) : same insertion order as plot_slice/ LUT
    """
    x_min, x_max = x_lims
    vy_min, vy_max = vy_lims

    # ---- 1. uniform grid -------------------------------------------------
    x_vec  = np.linspace(x_min, x_max, Nx)
    vy_vec = np.linspace(vy_min, vy_max, Ny)
    X, VY  = np.meshgrid(x_vec, vy_vec, indexing="ij")

    # ---- 2. analytic “inside” mask --------------------------------------
    vy_en  = np.sqrt(2*(E0 - U_tilde(x_vec, 0)))
    vy_p, vy_m, valid = crash_vy_branches(x_vec)

    mask   = np.zeros_like(X, dtype=bool)
    for i, xv in enumerate(x_vec):
        if np.isnan(vy_en[i]):  # outside energy envelope
            continue
        up_e, lo_e =  vy_en[i], -vy_en[i]
        if np.isnan(vy_p[i]):
            mask[i] = (VY[i] <= up_e) & (VY[i] >= lo_e)
        else:
            up_c, lo_c = vy_p[i], vy_m[i]
            mask[i] = ((VY[i] <= up_e) & (VY[i] >= lo_e) & (VY[i] >= up_c)) | \
                      ((VY[i] >= lo_e) & (VY[i] <= lo_c))

    # ---- 3. sample orbit points close to v_x = 0 -------------------------
    # --- 3.1 rebuild the SAME LUT that plot_slice builds 
    tab10 = [c for c in plt.cm.tab10.colors
             if mpl.colors.to_hex(c) != '#d62728']
    lut = {}
    # first assign colors for all_orb (skipping Moon crash)
    for x, vx, vy, lbl in all_orb:
        if lbl == "Moon crash ICs":
            continue
        lut.setdefault(lbl, tab10[len(lut) % len(tab10)])
    # then assign any new labels from all_cross
    for x, vx, vy, lbl in all_cross:
        lut.setdefault(lbl, tab10[len(lut) % len(tab10)])

    pts, labs = [], []
    for x, vx, vy, lbl in all_orb:
        if lbl == "Moon crash ICs":
            continue
        sel = np.abs(vx) <= vxtol
        pts.append(np.c_[(x[sel] - mu1) * x_scale, vy[sel]])
        labs += [lbl] * np.sum(sel)
    for x, vx, vy, lbl in all_cross:
        sel = np.abs(vx) <= vxtol
        pts.append(np.c_[(x[sel] - mu1) * x_scale, vy[sel]])
        labs += [lbl] * np.sum(sel)
    pts = np.vstack(pts)

    # ---- 4. KD-tree NN classification -----------------------------------
    fam_labels = []
    lut = {}                                    # insertion order = tab10 order
    for lbl in labs:
        if lbl not in lut:
            lut[lbl] = len(lut)
            fam_labels.append(lbl)
    colour_of_pt = np.array([lut[l] for l in labs], dtype=int)

    tree = KDTree(pts)

    flat_inside = np.where(mask.ravel())[0]
    query_xy    = np.column_stack([
        (X.ravel()[flat_inside] - mu1) * x_scale,
        VY.ravel()[flat_inside]
    ])

    colour_idx  = np.full(X.size, -1, dtype=int)
    _, nn       = tree.query(query_xy, k=1)
    colour_idx[flat_inside] = colour_of_pt[nn]

    # -------- 5. post-processing overrides (DRO ↔︎ DPO) -------------------
    # reshaping once makes the geometry tests a lot simpler
    colour_idx = colour_idx.reshape(X.shape)

    # helper: fill boolean masks row-wise (avoids shape mistakes)
    def _row_mask(row, cond):
        """Return a (Ny,) mask for the given row index."""
        m = np.zeros(Ny, dtype=bool)
        m[:] = cond
        return m

    # 5-a) retrograde override → map **every** retrograde cell to DRO
    if "Distant Retrograde" in fam_labels:
        dro_idx   = fam_labels.index("Distant Retrograde")
        retro_msk = np.zeros_like(mask, dtype=bool)

        for i in range(Nx):
            if not valid[i]:
                continue
            left  = (X[i] <  mu1) & (VY[i] >  vy_p[i])      # above crash curve
            right = (X[i] >  mu1) & (VY[i] <  vy_m[i])      # below crash curve
            retro_msk[i] = (left | right) & mask[i]

        colour_idx[retro_msk] = dro_idx

    # 5-b) prograde override → move **mis-labelled** DRO cells to DPO
    if {"Distant Retrograde", "Distant Prograde"}.issubset(fam_labels):
        dro_idx  = fam_labels.index("Distant Retrograde")
        dpo_idx  = fam_labels.index("Distant Prograde")
        pro_msk  = np.zeros_like(mask, dtype=bool)

        for i in range(Nx):
            if not valid[i]:
                continue
            left_pro  = (X[i] <  mu1) & (VY[i] < vy_m[i])   # below crash on LHS
            right_pro = (X[i] >  mu1) & (VY[i] > vy_p[i])   # above crash on RHS
            pro_msk[i] = (left_pro | right_pro) & mask[i]

        # only flip cells that are **currently** labelled as DRO
        override = pro_msk & (colour_idx == dro_idx)
        colour_idx[override] = dpo_idx

    # ---------------------------------------------------------------------
    return X, VY, mask, colour_idx, fam_labels, lut


def sample_truncated_gaussian_old(mean, cov, N, E0, exclude_crash: bool = True):
    """
    Generate N samples from a 2D Gaussian (x, vy) with given mean and covariance.
    If exclude_crash is True, reject any that lie in the crash region; otherwise
    include all points inside the energy envelope.

    Parameters
    ----------
    mean           : sequence of length 2, the Gaussian mean [x, vy]
    cov            : 2×2 covariance matrix
    N              : int, number of accepted samples to draw
    E0             : float, energy constant for envelope
    exclude_crash  : bool, if True reject samples in crash region

    Returns
    -------
    samples : (N, 2) array of accepted [x, vy] points
    """
    samples = []
    while len(samples) < N:
        # propose a batch
        M = max(1000, (N - len(samples)) * 5)
        pts = np.random.multivariate_normal(mean, cov, size=M)
        for x_i, vy_i in pts:
            # energy envelope
            vy_en = np.sqrt(2 * (E0 - U_tilde(x_i, 0)))
            if np.isnan(vy_en):
                continue

            # crash branches for this x
            vy_p, vy_m, valid = crash_vy_branches(np.array([x_i]))

            # decide if inside allowed region
            if exclude_crash:
                if np.isnan(vy_p):
                    # no crash surface here
                    inside = (-vy_en <= vy_i <= vy_en)
                else:
                    up_e, lo_e = vy_en, -vy_en
                    up_c, lo_c = vy_p[0], vy_m[0]
                    # accept only outside the crash interval
                    inside = ((up_c <= vy_i <= up_e) or
                              (lo_e <= vy_i <= lo_c))
            else:
                # include entire energy envelope, crash or not
                inside = (-vy_en <= vy_i <= vy_en)

            if inside:
                samples.append([x_i, vy_i])
            if len(samples) >= N:
                break

    return np.array(samples)

def sample_truncated_gaussian(
        mean,
        cov,
        N,
        E0,
        exclude_crash: bool = True,
        direction: str = "both",   # "prograde", "retrograde", or "both"
        ):
    """
    Draw N (x, v_y) samples from a 2-D Gaussian, restricted to the CR3BP
    energy envelope and (optionally) outside the crash region.  
    An extra *direction* flag lets you keep only
    prograde or retrograde points in the x–v_y plane.

    Parameters
    ----------
    mean, cov, N, E0, exclude_crash
        (unchanged – see original docstring)
    direction : {"prograde", "retrograde", "both"}
        Which side-of-motion region to keep.  Case-insensitive.
    mu        : float, optional
        The x-coordinate of the Moon (defaults to the global ``mu1``).

    Returns
    -------
    samples : (N, 2) ndarray
    """


    want_pro  = direction.lower() == "prograde"
    want_retro = direction.lower() == "retrograde"
    want_both = direction.lower() == "both"

    samples: list[list[float]] = []

    while len(samples) < N:
        M = max(1000, 5 * (N - len(samples)))
        pts = np.random.multivariate_normal(mean, cov, size=M)

        for x_i, vy_i in pts:
            # --- 1.  Energy envelope check -----------------------------------
            vy_en = np.sqrt(2 * (E0 - U_tilde(x_i, 0)))
            if np.isnan(vy_en) or not (-vy_en <= vy_i <= vy_en):
                continue  # outside Jacobi envelope

            # --- 2.  Crash-surface branches ----------------------------------
            vy_p, vy_m, valid = crash_vy_branches(np.array([x_i]))

            if exclude_crash and valid[0]:
                up_c, lo_c = vy_p[0], vy_m[0]
                if lo_c < vy_i < up_c:
                    continue  # inside crash funnel

            # --- 3.  Prograde / retrograde filter ----------------------------
            if not want_both and valid[0]:
                on_right = x_i > mu1
                on_left  = x_i < mu1
                above_pos = vy_i >= vy_p[0]
                below_neg = vy_i <= vy_m[0]

                if want_pro:
                    ok = (on_right and above_pos) or (on_left and below_neg)
                else:  # retrograde wanted
                    ok = (on_right and below_neg) or (on_left and above_pos)

                if not ok:
                    continue

            samples.append([x_i, vy_i])
            if len(samples) >= N:
                break

    return np.asarray(samples)

def sample_region_percentages(samples: np.ndarray,
                              classif_data: tuple[np.ndarray, ...],
                              x_scale: float = 1.0) -> np.ndarray:
    """
    Determine the percentage of (x,vy) samples falling into each region.

    Parameters
    ----------
    samples      : (N,2) array of [x, vy] points
    classif_data : (X, VY, mask, grid, fam_labels_per_idx, lut)
                   = output of classify_slice_region(...)
    fam_labels   : list of family names in the same order as classif_data
    x_scale      : scaling factor applied to x-distances in the KD-tree

    Returns
    -------
    percentages : (F+1,) array of percentages, where F = len(fam_labels)
                  the last entry is “Crash”.
    """
    # Unpack classification grid
    X, VY, mask, grid, fam_order, _ = classif_data

    # Build the same KD-tree on the *grid* that classify_slice_region used
    Xflat = X.ravel()
    VYflat = VY.ravel()
    Xscaled = (Xflat - mu1) * x_scale
    tree = KDTree(np.column_stack((Xscaled, VYflat)))
    idx_grid = grid.ravel()

    # Prepare classification array
    crash_label = len(fam_order)
    N = samples.shape[0]
    labels = np.full(N, crash_label, dtype=int)

    # Find which samples are finite
    finite_mask = np.isfinite(samples).all(axis=1)
    if finite_mask.any():
        good = samples[finite_mask]
        xg = (good[:,0] - mu1) * x_scale
        vyg = good[:,1]
        _, nn = tree.query(np.column_stack((xg, vyg)), k=1)
        idx_valid = idx_grid[nn]
        idx_valid[idx_valid < 0] = crash_label
        labels[finite_mask] = idx_valid

    # Count and convert to percentages
    F = len(fam_order) + 1   # +1 for Crash
    counts = np.bincount(labels, minlength=F)
    ratios = counts / counts.sum()

    return ratios

def calculate_posterior(ratio_calc_prior,measurement_ratios):
    normalization = np.sum(ratio_calc_prior * measurement_ratios)
    posterior = ratio_calc_prior * measurement_ratios / normalization
    return posterior