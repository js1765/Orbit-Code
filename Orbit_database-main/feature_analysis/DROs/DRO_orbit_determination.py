import numpy as np
from lineplot import LinePlot
from matplotlib import pyplot as plt
import pydylan


class Orbit_determination:

    def __init__(self):
        self.initial_state = [1.0306, 0.0, 0.0, 0.0, -0.0727, 0.0]
        self.orbital_period = 4.105528
        self.fuel_mass = 15000 #kg
        self.dry_mass = 10000 #kg
        self.initial_mass = self.fuel_mass + self.dry_mass
        self.initial_mass_norm = 1 # All masses are normalized with initial mass
        self.fuel_mass_norm = self.fuel_mass / self.initial_mass 
        self.dry_mass_norm = self.dry_mass / self.initial_mass
        self.thrust_acc = 0.199374249380613e-3 # in m^2/s
        self.thrust_norm_mass = self.initial_mass_norm * self.thrust_acc # Only normalized mass component for pydylan input
        self.Isp = 7365 
        self.number_of_segments = 200
        self.minimum_shooting_time = 35.0 # about 20 days
        self.maximum_shooting_time = 90.0 # about 50 days
        self.maximum_initial_coast_time = 0.0  
        self.maximum_final_coast_time = 5 # slightly more than period time of final orbit


    def get_cr3bp(self):    
        jupiter = pydylan.Body("Jupiter")
        europa = pydylan.Body("Europa")
        return pydylan.eom.CR3BP(primary=jupiter, secondary=europa)
    
    def get_phase_options(self):
        phase_options = pydylan.phase_options_structure()
        phase_options.number_of_segments = self.number_of_segments  
        phase_options.maximum_initial_coast_time = self.maximum_initial_coast_time 
        phase_options.maximum_final_coast_time = self.maximum_final_coast_time 
        phase_options.minimum_shooting_time = self.minimum_shooting_time 
        phase_options.maximum_shooting_time = self.maximum_shooting_time 
        phase_options.match_point_position_constraint_tolerance = 1E-3#1E-4
        phase_options.match_point_velocity_constraint_tolerance = 1E-3#1E-5
        phase_options.match_point_mass_constraint_tolerance = 1E-3#1E-3
        phase_options.control_coordinate_transcription = pydylan.enum.spherical
        phase_options.transcription = pydylan.enum.transcription_type.ForwardShooting
        return phase_options
    
    def get_orbit(self, state_on_orbit, orbit_period):
        cr3bp = self.get_cr3bp()
        thruster_parameters = pydylan.ThrustParameters(fuel_mass=self.fuel_mass_norm, dry_mass=self.dry_mass_norm, Isp=self.Isp, thrust=self.thrust_norm_mass)
        phase_options = self.get_phase_options()
        zero_control = np.array([0, orbit_period, 0] + [0, 0, 0] * self.number_of_segments + [self.fuel_mass_norm])

        mission_start = pydylan.FixedBoundaryCondition(state_on_orbit) 

        orbit = pydylan.Mission(cr3bp, mission_start, mission_start, pydylan.enum.snopt)  
        phase_options.optimal_control_methodology = pydylan.enum.optimal_control_methodology_type.Direct
        orbit.add_phase_options(phase_options)
        orbit.set_thruster_parameters(thruster_parameters)

        results_orbit = orbit.evaluate_and_return_solution(zero_control, maximum_integration_stepsize = 1E-3)
        return results_orbit

    def plot_orbits(self, orbit_states):
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel(r'$r_{1}$ (DU)', fontsize=12)
        ax.set_ylabel(r'$r_{2}$ (DU)', fontsize=12)

        ax.plot(orbit_states[:, 0], orbit_states[:, 1], label='Initial Orbit')



       
        fig.savefig("/Users/jannik/Documents/PhD_Princeton/Research/Auto_GNC/feature_analysis/DROs/dros/orbit.png")
        plt.close(fig)
        
        # from pathlib import Path
        # base_dir = Path(__file__).resolve().parent
        # fig_path = base_dir / "DROs" / "dros" / "orbit.png"
        # fig.savefig(fig_path, format='png', dpi=300)
        # plt.close(fig)



    def determine_orbit(self):
        orbit_states = self.get_orbit(self.initial_state,self.orbital_period).states
        self.plot_orbits(orbit_states)
        print(orbit_states[-1,:6])
        print(orbit_states[-1,:6]-self.initial_state)


if __name__ == '__main__':
    orbit = Orbit_determination()
    orbit.determine_orbit()