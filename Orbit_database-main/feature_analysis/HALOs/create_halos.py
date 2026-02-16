import pydylan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec



def get_cr3bp():    
    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    return pydylan.eom.CR3BP(primary=earth, secondary=moon)
    
def get_phase_options():
    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 20
    phase_options.match_point_position_constraint_tolerance = 1E-4
    phase_options.match_point_velocity_constraint_tolerance = 1E-5
    phase_options.match_point_mass_constraint_tolerance = 1E-3
    phase_options.control_coordinate_transcription = pydylan.enum.spherical
    return phase_options

def get_orbit(state_on_orbit,orbit_period):
    cr3bp = get_cr3bp()
    thruster_parameters = pydylan.ThrustParameters(fuel_mass=700, dry_mass=300, Isp=1000, thrust=1)
    phase_options = get_phase_options()
    zero_control = np.array([0, orbit_period, 0] + [0, 0, 0] * 20 + [700])

    mission_start = pydylan.FixedBoundaryCondition(state_on_orbit) 

    orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)  
    orbit.add_phase_options(phase_options)
    orbit.set_thruster_parameters(thruster_parameters)

    results_orbit = orbit.evaluate_and_return_solution(zero_control)
    return results_orbit

def plot(halo_states_list, energy_list):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.grid()
    ax.set_xlabel(r'$q_{2}$ (DU)', fontsize=12)
    ax.set_ylabel(r'$q_{3}$ (DU)', fontsize=12)
    #ax.set_title(r'Halo Orbits', fontsize=14)

    # Normalize energy_list to range from 0 to 1
    energy_min = min(energy_list)
    energy_max = max(energy_list)
    normalized_energy_list = [(e - energy_min) / (energy_max - energy_min) for e in energy_list]

    # Create a colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # Plot the halo states
    for i, halo_states in enumerate(halo_states_list):
        ax.plot(halo_states[:, 1], halo_states[:, 2], color=cmap(norm(normalized_energy_list[i])), label=f'Energy {energy_list[i]}')

    # Create the colorbar
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label(r'$\alpha$', fontsize=12)

    fig.savefig("/home/jg3607/Thesis/AAS_paper/results/boundary/halo_orbits/halo_orbits.pdf", format='pdf', dpi=300)
    plt.close(fig)

def plot_xy(halo_states_list, energy_list):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel(r'$q_{1}$ (DU)', fontsize=12)
    ax.set_ylabel(r'$q_{2}$ (DU)', fontsize=12)
    #ax.set_title(r'Halo Orbits', fontsize=14)

    # Normalize energy_list to range from 0 to 1
    energy_min = min(energy_list)
    energy_max = max(energy_list)
    normalized_energy_list = [(e - energy_min) / (energy_max - energy_min) for e in energy_list]

    # Create a colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # Plot the halo states
    for i, halo_states in enumerate(halo_states_list):
        ax.plot(halo_states[:, 0], halo_states[:, 1], color=cmap(norm(normalized_energy_list[i])), label=f'Energy {energy_list[i]}')

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$\alpha$', fontsize=12)

    fig.savefig("/home/jg3607/Thesis/AAS_paper/results/boundary/halo_orbits/halo_orbits_xy.pdf", format='pdf', dpi=300)
    plt.close(fig)


def plot_combined(halo_states_list, energy_list):
    
    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
            
    energy_min = min(energy_list)
    energy_max = max(energy_list)
    normalized_energy_list = [(e - energy_min) / (energy_max - energy_min) for e in energy_list]

    # Common properties
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # First subplot
    ax1.grid()
    ax1.set_xlabel(r'$q_{1}$ [DU]', fontsize=12)
    ax1.set_ylabel(r'$q_{2}$ [DU]', fontsize=12)
    for i, halo_states in enumerate(halo_states_list):
        ax1.plot(halo_states[:, 0], halo_states[:, 1], color=cmap(norm(normalized_energy_list[i])))

    # Second subplot
    ax2.grid()
    ax2.set_xlabel(r'$q_{3}$ [DU]', fontsize=12)
    ax2.tick_params(labelleft=False, left=False)  # Hide y-axis tick labels
    #ax2.set_yticklabels([])  # Hide y-axis tick labels
    for i, halo_states in enumerate(halo_states_list):
        ax2.plot(halo_states[:, 2], halo_states[:, 1], color=cmap(norm(normalized_energy_list[i])))

    # Create the colorbar
    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r'$\alpha$', fontsize=12)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12, wspace=0.1)
    # Save the figure
    fig.savefig("/Users/jannik/Documents/PhD_Princeton/Research/Auto_GNC/halo_orbits_L2/combined_halo_orbits.png", format='png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    cr3bp = get_cr3bp()
    libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    start = 0.008
    stop = 0.095
    step = 0.004
    #energys = np.arange(start, stop + step, step)
    energys = np.linspace(0.01, 0.05, 5)
    energy_list = energys.tolist()
    halo_states_list = []
    for halo_energy in energy_list:
        print(halo_energy)
        desired_orbit_energy = libration_point_information[1] + halo_energy
        halo = pydylan.periodic_orbit.Halo(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy, 40000.)
        print(halo.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success)
        halo_states = get_orbit(halo.orbit_initial_state,halo.orbit_period/2).states
        halo_states_list.append(halo_states)

    #plot(halo_states_list,energy_list)
    #plot_xy(halo_states_list,energy_list)
    plot_combined(halo_states_list,energy_list)