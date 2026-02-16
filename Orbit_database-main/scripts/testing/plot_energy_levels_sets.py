import numpy as np
import matplotlib.pyplot as plt
import pickle
import pydylan

# ------------------------------------------------------------------------------
# Example: Set up the CR3BP environment
# ------------------------------------------------------------------------------
earth = pydylan.Body('Earth')
moon = pydylan.Body('Moon')
cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

# Extract Earth and Moon radii in CR3BP dimensionless units
R_earth = earth.radius / cr3bp.DU
R_moon  = moon.radius  / cr3bp.DU

# Libration points info (L1, L2, etc.)
L1_info = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
L2_info = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)

# For demonstration, pick an energy E0 between L1 and L2 energies
E0 = L2_info[1]

# In many standard CR3BP notations:
mu1 = 1 - cr3bp.mu  # 'Earth' fraction if cr3bp.mu ~ mass fraction of Moon
mu2 = cr3bp.mu      # 'Moon' fraction

# Define the functions for the distances to the two primaries
def r1(x, y, mu2):
    return np.sqrt((x + mu2)**2 + y**2)

def r2(x, y, mu1):
    return np.sqrt((x - mu1)**2 + y**2)

# Define the effective (amended) potential function U_tilde.
def U_tilde(x, y, mu1, mu2):
    R1 = r1(x, y, mu2)
    R2 = r2(x, y, mu1)
    return -0.5*(x**2 + y**2) - mu1/R1 - mu2/R2 - 0.5*mu1*mu2

# Choose an x-range for the plot.
x = np.linspace(-1.2, 1.6, 500)

# Define a list of energy levels.
# In the CR3BP, these energies typically lie close to the energies at the Lagrange points.
# For example, L1 energy is around -1.594 and L2 around -1.505.
#energies = [L2_info[1], L2_info[1] - 0.5]
energies = [E0]#,[-2.209558146587854,-1.8815198737426182]
plt.figure(figsize=(8, 6))

# Loop over energy levels to plot the accessible region boundaries.
for E in energies:
    # Compute the potential along the x-axis (with y=0)
    U = U_tilde(x, 0, mu1, mu2)
    # Identify x-values where the kinetic energy term (E - U) is non-negative.
    valid = (E - U) >= 0
    # Calculate v_y (the velocity in the y direction) from the relation:
    # 1/2 * v_y^2 = E - U  =>  v_y = ± sqrt(2*(E-U))
    vy = np.sqrt(2*(E - U))
    # Plot both the positive and negative branches of v_y.
    plt.plot(x[valid], vy[valid] , label=f'E = {E:.3f}')
    plt.plot(x[valid], -vy[valid])




plt.xlabel('x')
plt.ylabel(r'$v_y$')
plt.xlim(-0.8, L2_info[0][0])
plt.ylim(-6, 6)
plt.title('Accessible Region in x–$v_y$ Plane for Different Energy Levels')
plt.legend()
plt.grid(True)
plt.show()