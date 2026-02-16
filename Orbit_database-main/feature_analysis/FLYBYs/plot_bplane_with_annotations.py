import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """
    Set equal scaling for 3D axes and center them at the origin (0,0,0).
    Ensures all axes (x, y, z) have the same scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    max_range = np.max([
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ])

    half_range = max_range / 2
    mid_x = (x_limits[0] + x_limits[1]) / 2
    mid_y = (y_limits[0] + y_limits[1]) / 2
    mid_z = (z_limits[0] + z_limits[1]) / 2
    ax.set_xlim3d(mid_x - half_range, mid_x + half_range)
    ax.set_ylim3d(mid_y - half_range, mid_y + half_range)
    ax.set_zlim3d(mid_z - half_range, mid_z + half_range)


def plot_b_plane(alpha, beta, gamma_range, distance_range):
    gamma_min, gamma_max = gamma_range
    d_min, d_max = distance_range

    # Normal vector of the B-plane
    n = np.array([
        -np.sin(alpha)*np.cos(beta),
        np.cos(alpha)*np.cos(beta),
        np.sin(beta)
    ])

    # Create a grid for the B-plane
    plane_size = 0.4  
    plane_limits = np.linspace(-plane_size, plane_size, 20)

    if not np.isclose(n[1], 0):
        X1_plane, X3_plane = np.meshgrid(plane_limits, plane_limits)
        X2_plane = (-n[0]*X1_plane - n[2]*X3_plane) / n[1]
    elif not np.isclose(n[0],0):
        X2_plane, X3_plane = np.meshgrid(plane_limits, plane_limits)
        X1_plane = (-n[1]*X2_plane - n[2]*X3_plane) / n[0]
    else:
        X1_plane, X2_plane = np.meshgrid(plane_limits, plane_limits)
        X3_plane = (-n[0]*X1_plane - n[1]*X2_plane) / n[2]

    # For alpha=0, beta=0:
    # x1 = D*cos(Gamma), x2=0, x3 = D*sin(Gamma)
    gamma_values = np.linspace(gamma_min, gamma_max, 50)
    d_values = np.linspace(d_min, d_max, 50)
    Gamma, D = np.meshgrid(gamma_values, d_values)
    x1 = D * np.cos(Gamma)
    x2 = np.zeros_like(x1)
    x3 = D * np.sin(Gamma)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the B-plane
    ax.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.3, color='blue', edgecolor='none')

    # Plot the target area
    ax.plot_surface(x1, x2, x3, color='red', alpha=0.8, edgecolor='none')

    # Mark origin
    ax.scatter(0, 0, 0, color='grey', s=50)

    # Draw a reference line along positive x1-axis
    ax.plot([0, d_max], [0, 0], [0, 0], color='black', linewidth=2)

    # Draw lines for gamma_min and gamma_max from origin out to d_max
    g_min_line_d = np.linspace(0, d_max, 100)
    x1_gmin = g_min_line_d * np.cos(gamma_min)
    x3_gmin = g_min_line_d * np.sin(gamma_min)
    ax.plot(x1_gmin, np.zeros_like(x1_gmin), x3_gmin, color='black', linestyle='--')

    g_max_line_d = np.linspace(0, d_max, 100)
    x1_gmax = g_max_line_d * np.cos(gamma_max)
    x3_gmax = g_max_line_d * np.sin(gamma_max)
    ax.plot(x1_gmax, np.zeros_like(x1_gmax), x3_gmax, color='black', linestyle='--')

    # Draw arcs to represent gamma_min and gamma_max angles
    arc_radius = 0.15
    # Arc for gamma_min
    arc_points_min = np.linspace(0, gamma_min, 30)
    x_arc_min = arc_radius * np.cos(arc_points_min)
    z_arc_min = arc_radius * np.sin(arc_points_min)
    ax.plot(x_arc_min, np.zeros_like(x_arc_min), z_arc_min, color='black')
    # Arc for gamma_max
    arc_points_max = np.linspace(0, gamma_max, 30)
    x_arc_max = arc_radius * np.cos(arc_points_max)
    z_arc_max = arc_radius * np.sin(arc_points_max)
    ax.plot(x_arc_max, np.zeros_like(x_arc_max), z_arc_max, color='black')

    # Label gamma_min and gamma_max near their arcs
    ax.text(x_arc_min[-1]+0.02, 0, z_arc_min[-1]+0.04, r'$\gamma_{\min}$', color='black', ha='right', va='bottom')
    ax.text(x_arc_max[-1]-0.04, 0, z_arc_max[-1]-0.08, r'$\gamma_{\max}$', color='black', ha='left', va='bottom')
    
    # d_min 
    ax.plot([d_min,d_min],[0,0],[0,0.02], color='green', linewidth=2)
    ax.text(d_min-0.04, 0, 0.02, r'$d_{\min}$', color='green', ha='center', va='bottom')

    # d_max 
    ax.plot([d_max,d_max],[0,0],[0,0.02], color='green', linewidth=2)
    ax.text(d_max+0.04, 0, 0.02, r'$d_{\max}$', color='green', ha='center', va='bottom')

    # Set axes equal
    set_axes_equal(ax)

    # Remove ticks but keep labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Set axis labels
    ax.set_xlabel(r'$r_{1}$')
    ax.set_ylabel(r'$r_{2}$')
    ax.set_zlabel(r'$r_{3}$')

    # Add label for B-plane at top-left corner of the figure
    ax.text(-0.3, 0, 0.2, 'B-plane', ha='left', va='top', fontsize=14, color='blue')
    ax.grid(True)
    # Adjust view
    ax.view_init(elev=60, azim=-90)
    ax.set_box_aspect([1,1,1])

    plt.show()


# Sample input values
alpha = 0
beta = 0
gamma_range = (-np.pi/4, np.pi/4)
distance_range = (0.25, 0.4)

plot_b_plane(alpha, beta, gamma_range, distance_range)