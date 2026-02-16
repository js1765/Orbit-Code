import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def set_axes_equal(ax):
    """
    Set equal scaling for 3D axes and center them at the origin (0, 0, 0).
    Ensures all axes (x, y, z) have the same scale for accurate visualization.

    Parameters:
    - ax: The matplotlib 3D axis object.
    """
    # Get axis limits
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
    """
    Plots the B-plane and highlights the target area defined by gamma and distance ranges.

    Parameters:
    - alpha: float, angle in radians.
    - beta: float, angle in radians.
    - gamma_range: tuple of floats (gamma_min, gamma_max), angles in radians.
    - distance_range: tuple of floats (d_min, d_max), distances.

    Returns:
    - None
    """

    # Convert angles from degrees to radians if necessary
    # alpha = np.deg2rad(alpha)
    # beta = np.deg2rad(beta)
    # gamma_range = (np.deg2rad(gamma_range[0]), np.deg2rad(gamma_range[1]))

    # Normal vector of the B-plane
    n = np.array([
        -np.sin(alpha) * np.cos(beta),
         np.cos(alpha) * np.cos(beta),
         np.sin(beta)
    ])

    # Create a grid for the B-plane
    plane_size = 0.4  # Adjust this for larger/smaller plane
    plane_limits = np.linspace(-plane_size, plane_size, 20)

    if not np.isclose(n[1], 0):  # Handle the case where n[2] is zero
        X1_plane, X3_plane = np.meshgrid(plane_limits, plane_limits)
        X2_plane = (-n[0] * X1_plane - n[2] * X3_plane) / n[1]  # The plane lies at X2 = 0
    elif not np.isclose(n[0], 0):
        X2_plane, X3_plane = np.meshgrid(plane_limits, plane_limits)
        X1_plane = (-n[1] * X2_plane - n[2] * X3_plane) / n[0]
    else:
        print("B-plane has a normal vector with n[2] != 0.")
        X1_plane, X2_plane = np.meshgrid(plane_limits, plane_limits)
        X3_plane = (-n[0] * X1_plane - n[1] * X2_plane) / n[2]

    # Generate target area points by varying gamma and distance
    gamma_min, gamma_max = gamma_range
    d_min, d_max = distance_range

    gamma_values = np.linspace(gamma_min, gamma_max, 50)
    d_values = np.linspace(d_min, d_max, 50)
    Gamma, D = np.meshgrid(gamma_values, d_values)

    # Compute x1, x2, x3
    x3 = D * np.sin(Gamma) * np.cos(beta)
    x1 = np.sin(alpha) * np.tan(beta) * x3 + D * np.cos(alpha) * np.cos(Gamma)
    x2 = -np.cos(alpha) * np.tan(beta) * x3 + D * np.sin(alpha) * np.cos(Gamma)

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the B-plane
    ax.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.3, color='blue', edgecolor='none')

    # Plot the target area
    ax.plot_surface(x1, x2, x3, color='red', alpha=0.8, edgecolor='none')

    ax.scatter(0, 0, 0, color='grey', s=400, depthshade=True)
    # Set labels and title
    set_axes_equal(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('B-plane with Target Area Highlighted')

    # Adjust the view angle for better visualization
    ax.view_init(elev=60, azim=-90)
    ax.set_box_aspect([1,1,1]) 

    plt.show()


# Sample input values
alpha = 0    # 45 degrees in radians
beta = 0       # 30 degrees in radians
gamma_range = (-np.pi/4, np.pi/4)  # From 0 to 90 degrees in radians
distance_range = (0.25, 0.4)       # From 1 to 5 units

# Call the function
plot_b_plane(alpha, beta, gamma_range, distance_range)