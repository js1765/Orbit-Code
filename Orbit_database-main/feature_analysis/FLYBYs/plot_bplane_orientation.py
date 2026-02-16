import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """
    Set equal scaling for 3D axes and center them at the origin (0, 0, 0).
    Ensures all axes (x, y, z) have the same scale for accurate visualization.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the range of each axis
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    # Find the maximum range
    max_range = max(x_range, y_range, z_range)

    # Set all axes to be centered at 0 with the same range
    half_range = max_range / 2
    ax.set_xlim3d([-half_range, half_range])
    ax.set_ylim3d([-half_range, half_range])
    ax.set_zlim3d([-half_range, half_range])

def plot_b_plane(alpha, beta, gamma_range, distance_range):
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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the B-plane
    ax.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.3, color='blue', edgecolor='none')

    # Mark the origin
    ax.scatter(0, 0, 0, color='grey', s=150, depthshade=True)

    line_width = 1.5
    ref_length = 0.9
    arc_radius = 0.3

    # A smaller arrow_length_ratio for less obtrusive arrowheads
    arrow_ratio = 0.05

    # Draw coordinate axes (r_1, r_2, r_3) with subtler arrowheads
    ax.quiver(0, 0, 0, ref_length, 0, 0, color='black', linewidth=line_width, arrow_length_ratio=arrow_ratio)
    ax.quiver(0, 0, 0, 0, ref_length, 0, color='black', linewidth=line_width, arrow_length_ratio=arrow_ratio)
    ax.quiver(0, 0, 0, 0, 0, ref_length, color='black', linewidth=line_width, arrow_length_ratio=arrow_ratio)

    # Label r_1, r_2, r_3
    ax.text(ref_length, 0, 0, r'$r_1$', color='black', ha='center', va='bottom', fontsize=16)
    ax.text(0, ref_length, 0, r'$r_2$', color='black', ha='center', va='bottom', fontsize=16)
    ax.text(0, 0, ref_length, r'$r_3$', color='black', ha='center', va='bottom', fontsize=16)

    # Normal vector line
    vec_length = 0.9
    ax.quiver(0, 0, 0, n[0]*vec_length, n[1]*vec_length, n[2]*vec_length,
              color='k', linewidth=line_width, arrow_length_ratio=arrow_ratio)
    # Label n
    ax.text(n[0]*vec_length, n[1]*vec_length, n[2]*vec_length, r'$n$', color='black',
            ha='center', va='bottom', fontsize=16)

    # Compute alpha arc
    alpha_val = alpha
    beta_val = beta

    alpha_points = np.linspace(0, alpha_val, 30)
    x_arc_alpha = arc_radius * (-np.sin(alpha_points))
    y_arc_alpha = arc_radius * (np.cos(alpha_points))
    z_arc_alpha = np.zeros_like(x_arc_alpha)
    ax.plot(x_arc_alpha, y_arc_alpha, z_arc_alpha, color='black', linewidth=line_width)
    ax.text(x_arc_alpha[-1]+0.07, y_arc_alpha[-1]+0.1, -0.1, r'$\alpha$', color='black',
            ha='center', va='bottom', fontsize=16)

    # Compute beta arc
    n_xy = np.array([n[0], n[1], 0])
    n_xy_norm = np.linalg.norm(n_xy)
    if n_xy_norm > 1e-9:
        n_xy_dir = n_xy / n_xy_norm
    else:
        n_xy_dir = np.array([0,1,0])

    beta_points = np.linspace(0, beta_val, 30)
    x_arc_beta_local = arc_radius * np.cos(beta_points)
    z_arc_beta_local = arc_radius * np.sin(beta_points)

    x_arc_beta_global = n_xy_dir[0]*x_arc_beta_local
    y_arc_beta_global = n_xy_dir[1]*x_arc_beta_local
    z_arc_beta_global = z_arc_beta_local

    ax.plot(x_arc_beta_global, y_arc_beta_global, z_arc_beta_global, color='black', linewidth=line_width)
    ax.text(x_arc_beta_global[-1], y_arc_beta_global[-1], z_arc_beta_global[-1], r'$\beta$', color='black',
            ha='center', va='bottom', fontsize=16)

    # Dotted line for the projection of n onto r_1-r_2 plane
    xy_line_length = 0.9
    ax.plot([0, n_xy_dir[0]*xy_line_length], [0, n_xy_dir[1]*xy_line_length], [0, 0],
            color='black', linestyle=':', linewidth=line_width)

    set_axes_equal(ax)

    # Remove all axis labeling and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')

    # Remove the background coordinate system
    ax.set_axis_off()

    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1,1,1])

    plt.show()


# Sample input values
alpha = np.pi/4    # 45 degrees
beta = np.pi/4     # 45 degrees
gamma_range = (-np.pi/4, np.pi/4)
distance_range = (0.25, 0.4)

plot_b_plane(alpha, beta, gamma_range, distance_range)