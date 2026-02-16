"""
plot.py — CR3BP plotting utilities (Earth–Moon, rotating frame)
===============================================================

Overview
--------
Plot helpers for visualizing CR3BP results: crash maps, Poincaré sections,
and (x, v_y, v_x) 3-D views. Designed to work with arrays and constants
from `support.constants` and helper routines from `support.helpers`.
No simulation logic here—only figure construction and saving.

Public functions (quick reference)
----------------------------------
• crash_map_2d(moon_hits, earth_hits, filename)
    2-D crash map in (x0, v_y0). Saves a PNG to `filename`.

• crash_map_3d_projection(data, filename)
    Projection of full 3-D crash catalogue onto (x0, v_y0). Saves a PNG.

• plot_2D_trajectory(trajectory, title=None, label=None, moon_plot=False)
    Simple planar trajectory plot (x vs y) with Earth/Moon marker.

• plot_collision_scatter_2D(crash_earth_x, crash_earth_vy, crash_moon_x, crash_moon_vy)
    2-D scatter of collision initial conditions by body.

• plot_collision_scatter_3D(crash_earth_x, crash_earth_vx, crash_earth_vy,
                             crash_moon_x,  crash_moon_vx,  crash_moon_vy)
    3-D scatter of collision initial conditions (x0, v_y0, v_x0).

• plot_crash_region(ax, *, x_range, vx_fixed, mu_body, R_body, x_body, color, label, ...)
    Draw analytic crash surface branches on y=0 section and shade crash region.

• plot_Poincare_2D(all_orbits, all_crossings, families_str, ... , which="vy")
    2-D Poincaré section (x vs v_y or v_x) with accessible envelope and crash regions.

• plot_Poincare_3D(all_orbits, all_crossings, families_str, ...)
    3-D Poincaré-style scatter in (x, v_y, v_x) with energy-envelope surface.

• plot_Poincare_analytic(all_orbits, all_crossings, res_num=10, show_families=False, retro=False)
    Analytic p:1 resonance curves overlaid on the accessible region and crash strips.

• plot_cross_section_x_vy_individual(..., vx_min=-6, vx_max=6, num_intervals=12, save_fig=False)
    Split the (x, v_y) plot into multiple figures, each for a v_x sub-interval.

• plot_cross_section_x_vy(..., vx_min, vx_max, num_intervals=12)
    Grid of subplots for (x, v_y) across v_x sub-intervals; saves a combined PNG.

Notes
-----
• File outputs are saved under `BASE_PATH` subfolders used by the caller.
• Internal helper `_add_bodies(ax)` draws Earth/Moon reference discs when constants exist.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from matplotlib.axes import Axes
from support.constants import cr3bp, mu1, mu2, R_earth, R_moon, earth_collision_radius, E0, jacobimin, L2_info, L1_info, U_tilde, BASE_PATH  # type: ignore
from support.helpers import earth_crash_vy_branches, generate_x_ranges, crash_surface_vy


HAS_BODIES = all(name in globals() for name in ("mu1", "mu2", "R_earth", "R_moon"))


# ------------------------------------------------------------------
#  Internal helper – draw discs
# ------------------------------------------------------------------

def _add_bodies(ax):
    if not HAS_BODIES:
        return
    # # ax.add_patch(Circle((-mu2, 0), R_earth, color="tab:green", alpha=0.3, lw=0))
    # # ax.add_patch(Circle((+mu1, 0), R_moon,  color="tab:gray",  alpha=0.3, lw=0))
    ax.scatter([-mu2], [0], s=200, c="tab:blue", alpha=0.3) #These sizes (the 's' values) are arbitrary to make them visible, they are not to scale or anything!
    ax.scatter([ mu1], [0], s=50,  c="tab:gray", alpha=0.3)
    # ax.scatter([-mu2], [0], s=200, marker="🌍", alpha=0.3) #These sizes (the 's' values) are arbitrary to make them visible, they are not to scale or anything!
    # ax.scatter([ mu1], [0], s=50,  marker="🌚", alpha=0.3)
    # ax.text(-mu2, 0, "🌍", fontsize=20, ha='center', va='center')
    # ax.text( mu1, 0, "🌙", fontsize=18, ha='center', va='center')


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

def crash_map_2d(moon_hits: List[Tuple[float, float]],
                 earth_hits: List[Tuple[float, float]],
                 filename: Union[str, Path]):
    fig, ax = plt.subplots(figsize=(6, 5))
    _add_bodies(ax)

    if moon_hits:
        xm, vym = np.array(moon_hits).T
        ax.scatter(xm, vym, s=6, color="royalblue",  label="Moon crash")
    if earth_hits:
        xe, vye = np.array(earth_hits).T
        ax.scatter(xe, vye, s=6, color="firebrick", label="Earth crash")

    ax.set_xlabel("$x_0$ [DU]")
    ax.set_ylabel("$v_{y0}$ [DU/TU]")
    ax.legend(loc="upper right")
    ax.set_title("CR3BP crash map (planar)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", lw=0.5)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def crash_map_3d_projection(data: List[Tuple[float, float, float, str]],
                            filename: Union[str, Path]):
    if not data:
        return

    xs, vxs, vys, bodies = zip(*data)
    xs, vys = np.array(xs), np.array(vys)
    bodies = np.array(bodies)

    fig, ax = plt.subplots(figsize=(6, 5))
    _add_bodies(ax)

    moon_mask  = bodies == "moon"
    earth_mask = bodies == "earth"
    ax.scatter(xs[moon_mask],  vys[moon_mask],  s=3, color="royalblue",  label="Moon crash")
    ax.scatter(xs[earth_mask], vys[earth_mask], s=3, color="firebrick", label="Earth crash")

    ax.set_xlabel("$x_0$ [DU]")
    ax.set_ylabel("$v_{y0}$ [DU/TU]")
    ax.legend(loc="upper right")
    ax.set_title("Full 3‑D crash catalogue (projection)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", lw=0.5)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)





def plot_2D_trajectory(trajectory, title =None, label=None, moon_plot=False): 
    plt.figure(figsize=(8,6))
    if trajectory.shape[1] == 7:
        x_traj = trajectory[:,1]
        y_traj = trajectory[:,2]
    else:
        x_traj = trajectory[:,0]
        y_traj = trajectory[:,1]
    plt.plot(x_traj, y_traj, label=label)
    # Mark Earth (secondary) and primary if desired
    earth_x = -mu2
    moon_x = mu1
    if moon_plot:
        plt.plot(moon_x, 0, 'bo', label="Moon")
    else:
        plt.plot(earth_x, 0, 'bo', label="Earth")
    # plt.plot(moon_x, 0, 'bo', label="Moon")
    # plt.plot(earth_x, 0, 'bo', label="Earth")
    # print(mu1)
    # print(mu2)
    # print(abs(mu1)+abs(mu2))
        

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.show()
    
    
    
    
    
####I ADDED THIS ONE:
def plot_2D_trajectory_earth_and_moon(trajectory, title =None, label=None): 
    plt.figure(figsize=(8,6))
    if trajectory.shape[1] == 7:
        x_traj = trajectory[:,1]
        y_traj = trajectory[:,2]
    else:
        x_traj = trajectory[:,0]
        y_traj = trajectory[:,1]
    plt.plot(x_traj, y_traj, label=label)
    # Mark Earth (secondary) and primary if desired
    earth_x = -mu2
    moon_x = mu1
    # print(mu1)
    plt.plot(moon_x, 0, 'bo', label="Moon")
    plt.plot(earth_x, 0, 'bo', label="Earth")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.show()
    
    
    
    
    
    
    

def plot_collision_scatter_2D(crash_earth_x, crash_earth_vy, crash_moon_x, crash_moon_vy):
    """
    Generates a 2D scatter plot of collision check results as specified.
    Displays Earth crashes (red) and Moon crashes (purple).
    """
    # -------------------------
    # Make a 3D scatter plot
    # -------------------------
    plt.figure(figsize=(9,7))

    # Earth collisions (red)
    plt.scatter(crash_earth_x, crash_earth_vy, color='red', s=10, label='Crash Earth')
    # Moon collisions (purple)
    plt.scatter(crash_moon_x, crash_moon_vy, color='purple', s=10, label='Crash Moon')

    plt.xlabel('x0')
    plt.ylabel('vy0')
    plt.title('2D CR3BP Crash Check')
    plt.legend()
    plt.show()

def plot_collision_scatter_3D(crash_earth_x, crash_earth_vx, crash_earth_vy, crash_moon_x,  crash_moon_vx,  crash_moon_vy):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')

    # Earth collisions (red)
    ax.scatter(crash_earth_x, crash_earth_vy, crash_earth_vx,
            color='red', s=2, label='Crash Earth')
    # Moon collisions (purple)
    ax.scatter(crash_moon_x, crash_moon_vy, crash_moon_vx,
            color='purple', s=2, label='Crash Moon')

    ax.set_xlabel('x0')
    ax.set_ylabel('vy0')
    ax.set_zlabel('vx0')
    ax.set_title('3D CR3BP Crash Check (Earth=red, Moon=purple)')
    ax.set_box_aspect((2, 1, 1))

    ax.legend()

# ------------------------------------------------------------------------------
# Poincare Section Plotting Functions
# ------------------------------------------------------------------------------
def plot_crash_region(
    ax: Axes,
    *,
    x_range: np.ndarray,
    vx_fixed: float,
    mu_body: float,
    R_body: float,
    x_body: float,
    color: str,
    label: str,
    alpha: float = 0.15,
    linewidth: float = 1.0,
    zorder: int = 0,
) -> None:
    """
    Draw the collision surface for a spherical body (Earth or Moon) on the
    Poincaré section y = 0 together with the energetically‑allowed crash region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x_range : ndarray
        1‑D grid of x values to test.
    vx_fixed : float
        Rotating‑frame v_x used when slicing the 3‑D surface.
    mu_body : float
        Gravitational parameter of the body (μₑ or μₘ).
    R_body : float
        Body radius nondimensionalised by Earth‑Moon distance.
    x_body : float
        x‑coordinate of body centre (−μ₂ for Earth, 1−μ₂ for Moon).
    E0, mu1, mu2, U_tilde
        Usual CR3BP quantities for the energy envelope.
    color, label, alpha, linewidth, zorder
        Plot styling.
    """


    # ------------------------------------------------------------------
    # 2) Crash‑surface branches from
    #  analytic formula
    # ------------------------------------------------------------------
    x_valid, vy_plus, vy_minus = crash_surface_vy(
        x_range, vx_fixed, mu_body, R_body, x_body
    )

    # Nothing to draw if the radicand was negative everywhere
    if x_valid.size == 0:
        return

    vy_env_valid = np.sqrt(2.0 * (E0 - U_tilde(x_valid, 0.0, mu1, mu2))- vx_fixed**2)
    

    # ------------------------------------------------------------------
    # 3) Boundary curves
    # ------------------------------------------------------------------
    mask_upper = np.abs(vy_plus) <= vy_env_valid
    mask_lower = np.abs(vy_minus) <= vy_env_valid

    ax.plot(x_valid[mask_upper], vy_plus[mask_upper], color=color, lw=linewidth, label=label)
    ax.plot(x_valid[mask_lower], vy_minus[mask_lower],color=color, lw=linewidth)

    # ------------------------------------------------------------------
    # 4) Shaded collision region (between ±v_y_env and the surface)
    # ------------------------------------------------------------------
    crash_mask = mask_upper | mask_lower
    if crash_mask.any():
        upper = np.minimum(vy_env_valid[crash_mask], vy_plus[crash_mask])
        lower = np.maximum(-vy_env_valid[crash_mask], vy_minus[crash_mask])
        ax.fill_between(
            x_valid[crash_mask], lower, upper,
            color=color, alpha=alpha, zorder=zorder
        )
        
        











############## def plot_energy_regions(trajectory, energy_min, energy_max, title =None, label=None, moon_plot=False):
    

        
        
        

def plot_Poincare_2D(all_orbits, all_crossings, families_str, 
                      earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                      plot_second_crossings=True, which="vy"):
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    x_min = -0.8
    x_max = L2_info[0][0]

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
            ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=2)
    

            
            

    already_labeled = set()
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            c = label_to_color.get(label, 'black')
            cross_label = label
            lbl = cross_label if cross_label not in already_labeled else None
            if lbl:
                already_labeled.add(cross_label)
            velocity_data = select_velocity(vx_vals, vy_vals)
            ax.scatter(x_vals, velocity_data, c=[c], marker='*', s=2, alpha=0.5, label=None)



    # Plot accessible region 
    x_en = np.linspace(x_min, x_max, 1000)
    vy_en = np.sqrt(2 * (E0 - U_tilde(x_en, 0, mu1, mu2)))
    ax.plot(x_en,  vy_en,  'k', lw=1, label="Accesible region")
    ax.plot(x_en, -vy_en,  'k', lw=1)
    
    
    #######IGNORE THIS PART--------------------------------------------------------------------------------------------
    # x_en = np.linspace(x_min, x_max, 1000)
    # # energy_levels = np.linspace(0, 1, 10)
    # # vy_en = np.sqrt(2 * (energy_levels*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # vy_en = np.sqrt(2 * (1.1*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # ax.plot(x_en,  vy_en,  'k', lw=1, label="1.1 E0")
    # ax.plot(x_en, -vy_en,  'k', lw=1)
    
    # vy_en = np.sqrt(2 * (1.3*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # ax.plot(x_en,  vy_en,  'k', lw=1, label="1.3 E0")
    # ax.plot(x_en, -vy_en,  'k', lw=1)
    
    # vy_en = np.sqrt(2 * (1.5*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # ax.plot(x_en,  vy_en,  'k', lw=1, label="1.5 E0")
    # ax.plot(x_en, -vy_en,  'k', lw=1)
    
    # vy_en = np.sqrt(2 * (1.7*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # ax.plot(x_en,  vy_en,  'k', lw=1, label="1.7 E0")
    # ax.plot(x_en, -vy_en,  'k', lw=1)
    
    # vy_en = np.sqrt(2 * (1.9*E0 - U_tilde(x_en, 0, mu1, mu2)))
    # ax.plot(x_en,  vy_en,  'k', lw=1, label="1.9 E0")
    # ax.plot(x_en, -vy_en,  'k', lw=1)
    #######-----------------------------------------------------------------------------------------------------------
    
    
    
    # ######THIS PART IS TO DISPLAY THE CURVES OF CONSTANT ENERGY ON THE DIAGRAM, IF YOU WANT TO SEE THOSE
    # for n in np.arange(1.0, 4 + 0.2, 0.2):
    #     vy_en = np.sqrt(2 * (n * E0 - U_tilde(x_en, 0, mu1, mu2)))
    #     ax.plot(x_en,  vy_en,  'k', lw=1, label=f"{n} E0")
    #     ax.plot(x_en, -vy_en,  'k', lw=1)
        
        
        
    

    # ------------------------------------------------------------------------------
    # Crash regions
    # ------------------------------------------------------------------------------
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]

    x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
    x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
    plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
    plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")



    # # print("E0 VALUE E0 VALUE E0 VALUE", E0)
    # # -1.5920953503382127


    # vy_env_valid = np.sqrt(2.0 * (-1.7 - U_tilde(x_valid, 0.0, mu1, mu2))- vx_fixed**2)

    # mask_upper = np.abs(vy_plus) <= vy_env_valid
    # mask_lower = np.abs(vy_minus) <= vy_env_valid

    # ax.plot(x_valid[mask_upper], vy_plus[mask_upper], color=color, lw=linewidth, label=label)
    # ax.plot(x_valid[mask_lower], vy_minus[mask_lower],color=color, lw=linewidth)
    
    
    





    if earth_crash_x0 is not None or earth_crash_vy0 is not None:
        ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
        ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')

    # #ax.set_xlim(0.0, L1_info[0][0])
    # ax.set_xlim(x_min, x_max)
    # #ax.set_ylim(-2, 6)
    # ax.set_ylim(-6, 6)
    
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-6, -0.5)
    
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    which_str = which if which in ["vx", "vy"] else "vy"
    plt.savefig(
        f"{BASE_PATH}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()













#------------------------------------------------------------------------------
######I WROTE THIS ONE:
def plot_Poincare_2D_with_balls(all_orbits, all_crossings, families_str, 
                      earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                      plot_second_crossings=True, which="vy", radius=0.01):

    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    _add_bodies(ax)
    x_min = -0.8
    x_max = L2_info[0][0]
    
    # ax.set_aspect('equal', adjustable='box')
    
    

    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker_idx = {}

    def select_velocity(vx_vals, vy_vals):
        return vy_vals if which == "vy" else vx_vals



    ### Helper function to draw circles ("balls") around points
    def draw_balls(ax, x_vals, y_vals, radius, colour):
        for x, y in zip(x_vals, y_vals):
            circ = Circle((x, y), radius,
                        edgecolor=colour,
                        facecolor=colour, 
                        # fill=1,
                        alpha=0.3,
                        linewidth=0.5)
            ax.add_patch(circ)
            



    ### Colours the points according to the family they belong to:
    for (x_vals, vx_vals, vy_vals, label) in all_orbits:
        if label not in label_to_color:
            label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
            label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
        c = label_to_color[label]
        m = marker_cycle[label_to_marker_idx[label]]
        velocity_data = select_velocity(vx_vals, vy_vals)
        if x_vals.size > 0:
            ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=2)  #Plots the first few points for some reason??
            draw_balls(ax, x_vals, velocity_data, radius, c)

            
            
    ### Plots all of the points after the initially plotted ones. (i.e., the "second" crossings, I think.)
    already_labeled = set()
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            c = label_to_color.get(label, 'black')
            cross_label = label
            lbl = cross_label if cross_label not in already_labeled else None
            if lbl:
                already_labeled.add(cross_label)
            velocity_data = select_velocity(vx_vals, vy_vals)
            ax.scatter(x_vals, velocity_data, c=[c], marker='*', s=2, alpha=0.5, label=None)
            draw_balls(ax, x_vals, velocity_data, radius, c)



    ### Plots the accessible region:
    x_en = np.linspace(x_min, x_max, 1000)
    vy_en = np.sqrt(2 * (E0 - U_tilde(x_en, 0, mu1, mu2)))
    ax.plot(x_en,  vy_en,  'k', lw=1, label="Accesible region")
    ax.plot(x_en, -vy_en,  'k', lw=1)
    
    

    
    
    ### Plot the crash regions
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]

    x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
    x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
    plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
    plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")

    if earth_crash_x0 is not None or earth_crash_vy0 is not None:
        ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
        ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')




    ###Sets the graph axes/dimensions, and saves and prints the figure:
    # ax.set_xlim(-0.8, L2_info[0][0])
    # ax.set_ylim(-6, 6)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-6, -0.5)
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    which_str = which if which in ["vx", "vy"] else "vy"
    plt.savefig(
        f"{BASE_PATH}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


#------------------------------------------------------------------------------











#------------------------------------------------------------------------------
######I WROTE THIS ONE:
def plot_Poincare_2D_with_boxes(all_orbits, all_crossings, families_str, 
                      earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                      plot_second_crossings=True, which="vy", square_length=0.01):

    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    _add_bodies(ax)
    x_min = -0.8
    x_max = L2_info[0][0]
    
    # ax.set_aspect('equal', adjustable='box')
    
    

    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker_idx = {}

    def select_velocity(vx_vals, vy_vals):
        return vy_vals if which == "vy" else vx_vals



    ### Helper function to draw circles ("balls") around points
    def draw_boxes(ax, x_vals, y_vals, square_length, colour):
        for x, y in zip(x_vals, y_vals):
            l = square_length/2
            rect = Rectangle((x - l/2, y-l/2), l, l,
                        edgecolor=colour,
                        facecolor=colour, 
                        # fill=1,
                        alpha=0.3,
                        linewidth=0.5)
            ax.add_patch(rect)
            
            


    ### Colours the points according to the family they belong to:
    for (x_vals, vx_vals, vy_vals, label) in all_orbits:
        if label not in label_to_color:
            label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
            label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
        c = label_to_color[label]
        m = marker_cycle[label_to_marker_idx[label]]
        velocity_data = select_velocity(vx_vals, vy_vals)
        if x_vals.size > 0:
            ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=2)  #Plots the first few points for some reason??
            draw_boxes(ax, x_vals, velocity_data, square_length, c)

            
            
    ### Plots all of the points after the initially plotted ones. (i.e., the "second" crossings, I think.)
    already_labeled = set()
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            c = label_to_color.get(label, 'black')
            cross_label = label
            lbl = cross_label if cross_label not in already_labeled else None
            if lbl:
                already_labeled.add(cross_label)
            velocity_data = select_velocity(vx_vals, vy_vals)
            ax.scatter(x_vals, velocity_data, c=[c], marker='*', s=2, alpha=0.5, label=None)
            draw_boxes(ax, x_vals, velocity_data, square_length, c)



    ### Plots the accessible region:
    x_en = np.linspace(x_min, x_max, 1000)
    vy_en = np.sqrt(2 * (E0 - U_tilde(x_en, 0, mu1, mu2)))
    ax.plot(x_en,  vy_en,  'k', lw=1, label="Accesible region")
    ax.plot(x_en, -vy_en,  'k', lw=1)
    
    

    
    
    ### Plot the crash regions
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]

    x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
    x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
    plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
    plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")

    if earth_crash_x0 is not None or earth_crash_vy0 is not None:
        ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
        ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')




    ###Sets the graph axes/dimensions, and saves and prints the figure:
    # ax.set_xlim(-0.8, L2_info[0][0])
    # ax.set_ylim(-6, 6)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-6, -0.5)
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    which_str = which if which in ["vx", "vy"] else "vy"
    plt.savefig(
        f"{BASE_PATH}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


#------------------------------------------------------------------------------







# #------------------------------------------------------------------------------
# ######I WROTE THIS ONE:
# def Poincare_2D_get_boxes(all_orbits, all_crossings, families_str, 
#                       earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                       plot_second_crossings=True, which="vy", square_length=0.01):

    
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111)
#     _add_bodies(ax)
#     x_min = -0.8
#     x_max = L2_info[0][0]
    
#     # ax.set_aspect('equal', adjustable='box')
    
    

#     color_cycle = plt.cm.tab10.colors
#     marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
#     label_to_color = {}
#     label_to_marker_idx = {}

#     def select_velocity(vx_vals, vy_vals):
#         return vy_vals if which == "vy" else vx_vals


#     boxes = []


#     ### Helper function to draw circles ("balls") around points
#     def add_box(x_vals, y_vals, square_length, colour):
#         for x, y in zip(x_vals, y_vals):
#             l = square_length/2
#             rect = Rectangle((x - l/2, y-l/2), l, l,
#                         edgecolor=colour,
#                         facecolor=colour, 
#                         # fill=1,
#                         alpha=0.3,
#                         linewidth=0.5)
#             # ax.add_patch(rect)
#             boxes.append(rect)
            
            


#     ### Colours the points according to the family they belong to:
#     for (x_vals, vx_vals, vy_vals, label) in all_orbits:
#         if label not in label_to_color:
#             label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
#             label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
#         c = label_to_color[label]
#         m = marker_cycle[label_to_marker_idx[label]]
#         velocity_data = select_velocity(vx_vals, vy_vals)
#         if x_vals.size > 0:
#             #ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=2)  #Plots the first few points for some reason??
#             add_box(x_vals, velocity_data, square_length, c)

            
            
#     ### Plots all of the points after the initially plotted ones. (i.e., the "second" crossings, I think.)
#     already_labeled = set()
#     if plot_second_crossings:
#         for (x_vals, vx_vals, vy_vals, label) in all_crossings:
#             c = label_to_color.get(label, 'black')
#             cross_label = label
#             lbl = cross_label if cross_label not in already_labeled else None
#             if lbl:
#                 already_labeled.add(cross_label)
#             velocity_data = select_velocity(vx_vals, vy_vals)
#             #ax.scatter(x_vals, velocity_data, c=[c], marker='*', s=2, alpha=0.5, label=None)
#             add_box(x_vals, velocity_data, square_length, c)



#     # ### Plots the accessible region:
#     # x_en = np.linspace(x_min, x_max, 1000)
#     # vy_en = np.sqrt(2 * (E0 - U_tilde(x_en, 0, mu1, mu2)))
#     # ax.plot(x_en,  vy_en,  'k', lw=1, label="Accesible region")
#     # ax.plot(x_en, -vy_en,  'k', lw=1)
    
    
#     for box in boxes:
#         ax.add_patch(box)
    


        
    
#     # ### Plot the crash regions
#     # x_min_moon=L1_info[0][0]
#     # x_max_moon=L2_info[0][0]
#     # x_min_earth=-0.8
#     # x_max_earth=L1_info[0][0]

#     # x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
#     # x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
#     # plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
#     # plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")

#     # if earth_crash_x0 is not None or earth_crash_vy0 is not None:
#     #     ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
#     #     ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')


    


#     ###Sets the graph axes/dimensions, and saves and prints the figure:
#     # ax.set_xlim(-0.8, L2_info[0][0])
#     # ax.set_ylim(-6, 6)
#     ax.set_xlim(0, 0.5)
#     ax.set_ylim(-6, -0.5)
    
#     ax.set_xlabel("x")
#     ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.tight_layout()
#     which_str = which if which in ["vx", "vy"] else "vy"
#     plt.savefig(
#         f"{BASE_PATH}/Figures/IC_plots/"
#         f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
#         dpi=300,
#         bbox_inches='tight'
#     )
#     plt.show()


# #------------------------------------------------------------------------------

















# ------------------------------------------------------------------------------
# 3D Plotting Function
# ------------------------------------------------------------------------------
def plot_Poincare_3D(all_orbits, all_crossings, families_str,
                    earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
                    plot_second_crossings=True):
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
    # print("L2_info[0][0] IS THE FOLLOWING", L2_info[0][0])
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





    ##########THIS IS THE PART THAT ACTUALLY PLOTS THE 3D SURFACE:
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
    ax.plot_surface(X, VY, VX, alpha=0.2, color='grey', edgecolor='none')
    
    
    ## print("HGFDJGKHJKGJFDHTXFGJCGVHKKGCFXGCJMVJ", E0)
    ## -1.5920953503382127
    ## print("HGFDJGKHJKGJFDHTXFGJCGVHKKGCFXGCJMVJ", cr3bp.mu)
    
    
    
    # ######TRYING TO PLOT A SIMILAR THING BUT FOR A LOWER ENERGY LEVEL (SO JUST THE SAME AS ABOVE BUT WITH A DIFFERNT E0):
    # x_arr  = np.linspace(x_min, x_max, N_x)
    # theta  = np.linspace(0, 2*np.pi, N_th)
    # X, TH = np.meshgrid(x_arr, theta)
    # mu1 = 1 - cr3bp.mu
    # mu2 = cr3bp.mu
    # U_vals = U_tilde(X, 0.0, mu1, mu2)
    # val    = 2.0*(0.9*E0 - U_vals)
    # val[val < 0] = np.nan
    # R = np.sqrt(val)
    # R_clipped = np.where(R > 6, 6, R)
    # VX = R_clipped * np.cos(TH)
    # VY = R_clipped * np.sin(TH)
    # ax.plot_surface(X, VY, VX, alpha=0.2, color='red', edgecolor='none')
    
    
    
    
    
    
    

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
        f"{BASE_PATH}/Figures/IC_plots/"
        f"IC_{families_str}_EM_CR3BP_x_vx_vy_2nd_{plot_second_crossings}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()






# ------------------------------------------------------------------------------
# 2‑D plotting function (analytic resonances only)
# ------------------------------------------------------------------------------
def plot_Poincare_analytic(all_orbits, all_crossings, res_num=10, show_families=False, retro=False):
    sign = 1
    if retro:
        sign *= -1  

    fig, ax = plt.subplots(figsize=(14, 8))

    x_min, x_max = -cr3bp.mu + 2 * R_earth, L1_info[0][0]
    x_plus = np.linspace(x_min, x_max, 600)
    r_plus = np.abs(x_plus + mu2)

    # ── 1) accessible‑region envelope ────────────────────────────────────────
    vy_env = np.sqrt(2 * (E0 - U_tilde(x_plus, 0.0, mu1, mu2)))
    ax.plot(x_plus,  vy_env, color="black", label="Accessible region")
    ax.plot(x_plus, -vy_env, color="black")

    # ── pre‑compute crash region boundaries (Earth) on same x‑grid ───────────
    vy_p_e, vy_m_e, valid_e = earth_crash_vy_branches(x_plus)
    crash_strip = valid_e & (
        (np.abs(vy_p_e) <= vy_env) | (np.abs(vy_m_e) <= vy_env)
    )
    upper_e = np.minimum(vy_env, vy_p_e)
    lower_e = np.maximum(-vy_env, vy_m_e)

    # ── 2) two‑body circular‑orbit approximation ────────────────────────────
    vy_circ = sign * np.sqrt(mu1 / r_plus) - (x_plus + mu2)
    valid_acc  = np.abs(vy_circ) <= vy_env
    in_crash   = crash_strip & (vy_circ >= lower_e) & (vy_circ <= upper_e)
    keep_circ  = valid_acc & ~in_crash
    #ax.plot(x_plus[keep_circ], vy_circ[keep_circ], color="green", label="2‑body circular")

    # ── 3) p:1 resonant approximations (p = 2 … res_num) ─────────────────────
    cmap = plt.cm.plasma(np.linspace(0.05, 0.95, res_num))
    for idx, p in enumerate(range(2, res_num)):
        a_p   = (mu1 / p) ** (2 / 3)
        radic = 2 * mu1 * (1.0 / r_plus - 1.0 / (2.0 * a_p))
        base  = radic >= 0.0                       # real roots only
        if not base.any():
            continue

        vy_res = sign * np.sqrt(radic[base]) - x_plus[base]

        # ── accessible‑region filter ────────────────────────────────────────────
        good_acc = np.abs(vy_res) <= vy_env[base]

        # ── crash‑region veto ───────────────────────────────────────────────────
        in_crash = (
            crash_strip[base]
            & (vy_res >= lower_e[base])
            & (vy_res <= upper_e[base])
        )

        keep = good_acc & ~in_crash
        if not keep.any():
            continue

        # ── split into contiguous segments so gaps aren’t connected ─────────────
        keep_idx = np.where(keep)[0]                    # indices inside *base*
        # split where index jumps by more than 1
        splits = np.where(np.diff(keep_idx) > 1)[0] + 1
        segments = np.split(keep_idx, splits)

        for j, seg in enumerate(segments):
            if seg.size == 0:
                continue
            x_seg  = x_plus[base][seg]
            vy_seg = vy_res[seg]
            # label only first segment so legend isn’t duplicated
            label = f"{p}:1 resonant approximation" if j == 0 else None
            ax.plot(x_seg, vy_seg, color=cmap[idx], lw=1, linestyle="--", label=label)

        # optional: report overall x‑range of all valid points
        x_valid = x_plus[base][keep]
        print(f"{p}:1 resonant approximation valid x‑range: "
            f"[{x_valid.min():.4f}, {x_valid.max():.4f}]")
        
        # Plot bifurcation points
        i = p
        r = (1/i)**(2/3)
        x = r - mu2
        #ax.scatter(r - mu2, sign*np.sqrt(mu1 / r) - (x + mu2), color=cmap[idx])
    
    if show_families:
        color_cycle = plt.cm.tab10.colors
        marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
        label_to_color = {}
        label_to_marker_idx = {}

        
        # Plot orbits
        for (x_vals, vx_vals, vy_vals, label) in all_orbits:
            if label not in label_to_color:
                label_to_color[label] = color_cycle[len(label_to_color) % len(color_cycle)]
                label_to_marker_idx[label] = len(label_to_marker_idx) % len(marker_cycle)
            c = label_to_color[label]
            m = marker_cycle[label_to_marker_idx[label]]
            velocity_data = vy_vals
            if x_vals.size > 0:
                ax.scatter(x_vals, velocity_data, c=[c], marker=m, label=label, s=5)
    
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]

    x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
    x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
    plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
    plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")

    # ── styling & save ───────────────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    vy_min = min(sign*-2, sign*6)
    vy_max= max(sign*-2, sign*6)
    ax.set_ylim(vy_min, vy_max)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$")
    if res_num < 20:
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/Figures/IC_plots/IC_resonances_analytic_num_{res_num}_orbs_{show_families}_retro_{retro}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

def plot_cross_section_x_vy_individual(all_orbits, all_crossings, families_id,
                                        earth_crash_x0, earth_crash_vx0, earth_crash_vy0,
                                        moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
                                        vx_min = -6, vx_max = 6, num_intervals=12, save_fig=False):
    """
    Generates individual 2D scatter plots of (x, v_y) for each subinterval in the vx direction.
    Each interval is saved or displayed as a separate figure.

    Parameters:
        all_orbits:          List of tuples (x_vals, vx_vals, vy_vals, label) for each orbit.
        all_crossings:       List of tuples (x_vals, vx_vals, vy_vals, label) for crossing points.
        vx_min:              Minimum vx value.
        vx_max:              Maximum vx value.
        earth_crash_x0:      x-values of Earth crash points
        earth_crash_vx0:     vx-values of Earth crash points
        earth_crash_vy0:     vy-values of Earth crash points
        moon_crash_x0:       x-values of Moon crash points
        moon_crash_vx0:      vx-values of Moon crash points
        moon_crash_vy0:      vy-values of Moon crash points
        num_intervals:       Number of vx intervals to split the data into.
        save_fig:            If True, each figure is saved to disk instead of just shown.
        fig_prefix:          Prefix for figure filenames (when save_fig=True).
    """
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]
    
    # 1) Compute vx interval boundaries.
    interval_edges = np.linspace(vx_min, vx_max, num_intervals + 1)

    # 2) Setup color/marker cycles so that the same orbit label gets the same color/marker
    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker = {}

    # 3) Loop over intervals, creating one figure per interval
    for i in range(num_intervals):
        lower_bound, upper_bound = interval_edges[i], interval_edges[i+1]

        # Create a new figure
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)

        # For each new figure, we reset "crash" labels so that we can show them if needed
        crash_earth_plotted = False
        crash_moon_plotted = False

        # 4) Plot orbits
        for (x_vals, vx_vals, vy_vals, label) in all_orbits:
            if label not in label_to_color:
                idx = len(label_to_color) % len(color_cycle)
                label_to_color[label] = color_cycle[idx]
                label_to_marker[label] = marker_cycle[idx]

            mask = (vx_vals >= lower_bound) & (vx_vals <= upper_bound)
            if np.any(mask):
                plt.scatter(x_vals[mask], vy_vals[mask],
                            color=label_to_color[label],
                            marker=label_to_marker[label],
                            s=1, alpha=1.0, label=label)

        # 5) Plot crossing points
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            mask = (vx_vals >= lower_bound) & (vx_vals <= upper_bound)
            if np.any(mask):
                # If the crossing label wasn't used among orbits, default to black
                plt.scatter(x_vals[mask], vy_vals[mask],
                            color=label_to_color.get(label, 'black'),
                            marker='*', s=1, alpha=1.0, label=label)

        # 6) Plot Earth crash points (if any)
        mask_earth = (earth_crash_vx0 >= lower_bound) & (earth_crash_vx0 <= upper_bound)
        if np.any(mask_earth):
            # Use a label only if it's the first time for this figure
            label_val = 'Crash into Earth' if not crash_earth_plotted else None
            plt.scatter(np.array(earth_crash_x0)[mask_earth],
                        np.array(earth_crash_vy0)[mask_earth],
                        color='red', s=0.2, alpha=0.2, label=label_val)
            crash_earth_plotted = True

        # 7) Plot Moon crash points (if any)
        mask_moon = (moon_crash_vx0 >= lower_bound) & (moon_crash_vx0 <= upper_bound)
        if np.any(mask_moon):
            label_val = 'Crash into Moon' if not crash_moon_plotted else None
            plt.scatter(np.array(moon_crash_x0)[mask_moon],
                        np.array(moon_crash_vy0)[mask_moon],
                        color='purple', s=0.2, alpha=0.2, label=label_val)
            crash_moon_plotted = True

        # 8) Plot the accessible region (example below assumes same variables as your code)
        # Make sure these variables (cr3bp, R_earth, etc.) are in scope or passed to this function

        vx_fixed = min(abs(lower_bound), abs(upper_bound))

        x_plus  = np.linspace(-cr3bp.mu + 2*R_earth, x_max_moon, 10000)
        y_plus  = np.sqrt(2*(E0 - U_tilde(x_plus, 0, mu1, mu2)) - vx_fixed**2)
        x_minus = np.linspace(-cr3bp.mu - 2*R_earth, x_min_earth, 10000)
        y_minus = np.sqrt(2*(E0 - U_tilde(x_minus, 0, mu1, mu2)) - vx_fixed**2)
        
        # Positive and negative branches
        plt.plot(x_minus,  y_minus,  color='black', label="Accessible region")
        plt.plot(x_minus, -y_minus, color='black')
        plt.plot(x_plus,   y_plus,   color='black')
        plt.plot(x_plus,  -y_plus,   color='black')

        # 9) Final formatting for this figure
        plt.xlim(-0.8, L2_info[0][0])
        plt.ylim(-6, 6)
        plt.xlabel("x")
        plt.ylabel(r"$v_y$")
        plt.title(rf"$v_x \in [{lower_bound:.2f}, {upper_bound:.2f}]$")
        plt.grid(True)

        x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
        x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)

        # --- Moon ---
        plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=vx_fixed, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Moon crash surface")

        # --- Earth ---
        plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=vx_fixed, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Earth crash surface")

        # Combine handles and create a figure-level legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # We can remove duplicates if desired:
        unique = list(dict(zip(labels, handles)).items())  # or more robust approach
        plt.legend([x[1] for x in unique], [x[0] for x in unique],
                   loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.subplots_adjust(right=0.85)
        # 10) Save or show the figure
        if save_fig:
            plt.savefig(
            f"{BASE_PATH}/Figures/IC_plots_cross_sections/"
            f"IC_{families_id}_EM_CR3BP_x_vy_section_vx_{lower_bound:.1f}_{upper_bound:.1f}.png",
            dpi=300,
            bbox_inches='tight'
        )
        else:
            plt.show()

def plot_cross_section_x_vy(all_orbits, all_crossings, families_id, 
                            earth_crash_x0, earth_crash_vx0, earth_crash_vy0,
                            moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
                            vx_min, vx_max, num_intervals=12):
    """
    Generates 2D scatter plots of (x, v_y) for each subinterval in the vx direction.
    In each subplot, orbits, crossing points, and crash points are filtered to include
    only those data where vx is within the current subinterval.
    
    Parameters:
        all_orbits: List of tuples (x_vals, vx_vals, vy_vals, label) for each orbit.
        all_crossings: List of tuples (x_vals, vx_vals, vy_vals, label) for crossing points.
        vx_min: Minimum vx value.
        vx_max: Maximum vx value.
        earth_crash_x0, earth_crash_vx0, earth_crash_vy0: Crash point data for Earth.
        moon_crash_x0, moon_crash_vx0, moon_crash_vy0: Crash point data for Moon.
        num_intervals: Number of vx intervals to split the data into.
    """
    # Compute vx interval boundaries.
    interval_edges = np.linspace(vx_min, vx_max, num_intervals + 1)
    
    # Setup a grid of subplots.
    n_cols = 4  # For 10 intervals, a grid of 2 rows x 5 columns works well.
    n_rows = int(np.ceil(num_intervals / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Consistent color/marker cycles for orbits.
    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', '^', 's', 'd', 'v', '>', '<', 'p', 'h', 'x']
    label_to_color = {}
    label_to_marker = {}
    
    # Flags to avoid duplicate crash point labels in the legend.
    crash_earth_plotted = False
    crash_moon_plotted = False
    
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]
    # Loop over each vx subinterval.
    for i in range(num_intervals):
        lower_bound, upper_bound = interval_edges[i], interval_edges[i+1]
        ax = axes[i]
        ax.set_xlim(x_min_earth, x_max_moon)
        ax.set_ylim(-6, 6)
        # Plot orbits.
        for (x_vals, vx_vals, vy_vals, label) in all_orbits:
            if label not in label_to_color:
                idx = len(label_to_color) % len(color_cycle)
                label_to_color[label] = color_cycle[idx]
                label_to_marker[label] = marker_cycle[idx]
            
            mask = (vx_vals >= lower_bound) & (vx_vals <= upper_bound)
            if np.any(mask):
                ax.scatter(x_vals[mask], vy_vals[mask],
                           color=label_to_color[label],
                           marker=label_to_marker[label],
                           s=1,
                           alpha=1.0,
                           label=label)
        
        # Plot crossing points.
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            mask = (vx_vals >= lower_bound) & (vx_vals <= upper_bound)
            if np.any(mask):
                ax.scatter(x_vals[mask], vy_vals[mask],
                           color=label_to_color.get(label, 'black'),
                           marker='*',
                           s=1,
                           alpha=1.0,
                           label=f"{label} crossing")
        
        # Plot Earth crash points if they fall within the current vx subinterval.
        mask_earth = (earth_crash_vx0 >= lower_bound) & (earth_crash_vx0 <= upper_bound)
        if np.any(mask_earth):
            label_val = 'Crash into Earth' if not crash_earth_plotted else None
            ax.scatter(np.array(earth_crash_x0)[mask_earth],
                       np.array(earth_crash_vy0)[mask_earth],
                       color='red', s=0.2, alpha=0.1, label=label_val)
            crash_earth_plotted = True
        
        # Plot Moon crash points if they fall within the current vx subinterval.
        mask_moon = (moon_crash_vx0 >= lower_bound) & (moon_crash_vx0 <= upper_bound)
        if np.any(mask_moon):
            label_val = 'Crash into Moon' if not crash_moon_plotted else None
            ax.scatter(np.array(moon_crash_x0)[mask_moon],
                       np.array(moon_crash_vy0)[mask_moon],
                       color='purple', s=0.2, alpha=0.1, label=label_val)
            crash_moon_plotted = True
        
        # Plot accessible region 
        vx_fixed = min(abs(lower_bound),abs(upper_bound))
        x_plus = np.linspace(-cr3bp.mu + 2*R_earth, x_max_moon, 10000)
        y_dot_plus_plus  = np.sqrt(2*(E0 - U_tilde(x_plus, 0, mu1, mu2)) - vx_fixed**2)
        y_dot_plus_minus = -np.sqrt(2*(E0 - U_tilde(x_plus, 0, mu1, mu2)) - vx_fixed**2)
        x_minus = np.linspace(-cr3bp.mu - 2*R_earth, x_min_earth, 10000)
        y_dot_minus_plus  = np.sqrt(2*(E0 - U_tilde(x_minus, 0, mu1, mu2)) - vx_fixed**2)
        y_dot_minus_minus = -np.sqrt(2*(E0 - U_tilde(x_minus, 0, mu1, mu2)) - vx_fixed**2)

        ax.plot(x_minus, y_dot_minus_plus,  color='black', label="Accessible region")
        ax.plot(x_minus, y_dot_minus_minus, color='black')
        ax.plot(x_plus,  y_dot_plus_plus,   color='black')
        ax.plot(x_plus,  y_dot_plus_minus,  color='black')
        
        x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
        x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)

        # --- Moon ---
        plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=vx_fixed, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Moon crash surface")

        # --- Earth ---
        plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=vx_fixed, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Earth crash surface")

        ax.set_title(fr"$v_x$ ∈ [{lower_bound:.2f}, {upper_bound:.2f}]")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$v_y$")
        ax.grid(True)
    
    # Remove any unused subplots.
    for j in range(num_intervals, len(axes)):
        fig.delaxes(axes[j])
    
    # Create a global legend by combining handles.
    handles, labels = axes[0].get_legend_handles_labels()
    # Add dummy handles for crash points if they were plotted.
    if crash_earth_plotted:
        dummy_earth = plt.Line2D([], [], marker='o', color='red',
                                 linestyle='None', markersize=5, label='Crash into Earth')
        handles.append(dummy_earth)
        labels.append('Crash into Earth')
    if crash_moon_plotted:
        dummy_moon = plt.Line2D([], [], marker='o', color='purple',
                                linestyle='None', markersize=5, label='Crash into Moon')
        handles.append(dummy_moon)
        labels.append('Crash into Moon')
    
    #fig.legend(handles, labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(
        f"{BASE_PATH}/Figures/IC_plots_cross_sections/"
        f"IC_{families_id}_EM_CR3BP_x_vy_{num_intervals}_sections_vx_{vx_min}_{vx_max}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    
    
    
    
    
    
    

#------------------------------------------------------------------------------
######I WROTE THIS ONE:
def plot_given_boxes(boxes, earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, which="vy"):
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    _add_bodies(ax)
    x_min = -0.8
    x_max = L2_info[0][0]
    # print("AAAAAAAAAAAAAA")
    
    def select_velocity(vx_vals, vy_vals):
        return vy_vals if which == "vy" else vx_vals
    
#     print("BBBBBBBBBBBBBB")


    # for box in boxes:
    #     print("BOX IS THE FOLLOWING", box)

    # print("CCCCCCCCCC")
    

    for box in boxes:
        # print("BOX IS THE FOLLOWING", box)
        ax.add_patch(box)
    
    # print("DDDDDDDDDD")
    
    
    ### Plots the accessible region:
    x_en = np.linspace(x_min, x_max, 1000)
    vy_en = np.sqrt(2 * (E0 - U_tilde(x_en, 0, mu1, mu2)))
    ax.plot(x_en,  vy_en,  'k', lw=1, label="Accesible region")
    ax.plot(x_en, -vy_en,  'k', lw=1)
      
      
    
    ### Plot the crash regions
    x_min_moon=L1_info[0][0]
    x_max_moon=L2_info[0][0]
    x_min_earth=-0.8
    x_max_earth=L1_info[0][0]

    x_moon  = generate_x_ranges(x_min_moon,  x_max_moon, mu1, R_moon)
    x_earth = generate_x_ranges(x_min_earth, x_max_earth, -mu2, earth_collision_radius)
    plot_crash_region(ax=ax, x_range=x_moon, vx_fixed=0.0, mu_body=mu2, R_body=R_moon, x_body=1.0 - mu2, color="purple", label="Crash region Moon")
    plot_crash_region(ax=ax, x_range=x_earth, vx_fixed=0.0, mu_body=mu1, R_body=earth_collision_radius, x_body=-mu2, color="red", label="Crash region Earth")

    if earth_crash_x0 is not None or earth_crash_vy0 is not None:
        ax.scatter(earth_crash_x0, select_velocity(earth_crash_vx0, earth_crash_vy0), s=5, alpha = 0.005, color='red',    label='Crash into Earth')
        ax.scatter(moon_crash_x0,  select_velocity(moon_crash_vx0, moon_crash_vy0),  s=5, alpha = 0.005, color='purple', label='Crash into Moon')


    
     ###Sets the graph axes/dimensions, and saves and prints the figure:
    # ax.set_xlim(-0.8, L2_info[0][0])
    # ax.set_ylim(-6, 6)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-6, -0.5)
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_y$" if which == "vy" else r"$v_x$")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # which_str = which if which in ["vx", "vy"] else "vy"
    # plt.savefig(
    #     f"{BASE_PATH}/Figures/IC_plots/"
    #     f"IC_{families_str}_EM_CR3BP_x_{which_str}_2nd_{plot_second_crossings}.png",
    #     dpi=300,
    #     bbox_inches='tight'
    # )
    plt.show()

#------------------------------------------------------------------------------
