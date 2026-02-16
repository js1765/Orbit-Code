import pydylan
import numpy as np
import matplotlib.pyplot as plt




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

def plot_xy(halo_states_list, energy_list):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel(r'$r_{1}$ (DU)', fontsize=12)
    ax.set_ylabel(r'$r_{2}$ (DU)', fontsize=12)
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
        ax.plot(halo_states[:, 0], halo_states[:, 1], color=cmap(norm(normalized_energy_list[i])), label=f'x0 {energy_list[i]}')

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$x_0$', fontsize=12)
    plt.show()
    #fig.savefig("/Users/jannik/Documents/PhD_Princeton/Research/Auto_GNC/feature_analysis/DROs/plots/dros_xy.png", format='png', dpi=300)
    #plt.close(fig)


if __name__ == "__main__":

    cr3bp = get_cr3bp()
    start = 0.3
    stop = 0.5
    step = 0.01
    x_starts = np.arange(start, stop, step)
    x_start_list = x_starts.tolist()
    reso_states_list = []
    x_start_list = [0.45]  # 
    res_orbit_options = pydylan.periodic_orbit.ResonanceOptions()
    res_orbit_options.p = 7.0
    res_orbit_options.q = 1.0
    res_orbit_options.retrograde = True
    for reso_x in x_start_list:
        res_orbit_options.x = reso_x
        reso_orbit = pydylan.periodic_orbit.Resonance(cr3bp,res_orbit_options)
        assert reso_orbit.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success 
        #reso_orbit_states = get_orbit(reso_orbit.orbit_initial_state,reso_orbit.orbit_period).states
        #reso_states = get_orbit(reso_orbit.orbit_initial_state,reso_orbit.orbit_period).states
        reso_states_list.append(reso_orbit.orbit_state_history)
        print(reso_orbit.get_stability_index())
        print(reso_orbit.orbit_initial_state)
        print(reso_orbit.orbit_energy)
        print(reso_orbit.orbit_energy*-2)

    #plot_xy(reso_states_list,x_start_list)
