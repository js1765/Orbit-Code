"""
CR3BP dataset loader & plotting driver
======================================

Purpose
-------
Loads orbit/crossing datasets for Earth–Moon CR3BP visualizations and calls one of
several plotting routines. You can toggle *what* to load via the boolean flags
passed to `gather_dataset(...)`, and then choose a plotting function by
uncommenting it near the bottom.

Arguments to gather_dataset(...)
-------------------------------
plot_second_crossings : bool
    If True, compute/store ALL y=0 crossing points for the loaded orbits;
    if False, only initial (x0, vx0, vy0) are returned ==> this is way faster since no orbit propagation is requiured.

load_lunar : bool
    Load lunar families (Lyapunov L1, Distant Retrograde, Distant Prograde)
    from the JPL periodic-orbits API.

load_prograde_resonant : bool
    Load prograde resonant families.

load_prograde_resonant_x1 : bool
    Load only prograde X:1 resonant branches (i.e., ratio = 1.0 in filenames).

load_retrograde_resonant : bool
    Load retrograde resonant families.

load_retrograde_resonant_x1 : bool
    Load only retrograde X:1 resonant branches (i.e., ratio = 1.0 in filenames).

load_crash : bool
    Load crash initial conditions (Earth/Moon) from local pickle files (x0, vx0, vy0).

load_circular : bool
    Load locally stored two-body circular-orbit seeds (converted to Jacobi energy),
    useful as reference curves.

Notes
-----
• The returned dict includes: "orbits", "crossings" (if requested), crash IC arrays,
  and "families_str" (a compact label of what was loaded).
• You can also build an analytic resonant dataset via `gather_analytic_resonant(...)`
  and plot those curves instead of (or in addition to) catalog data.
"""


# from support.orbit_data import merge_singleton_datasets, gather_analytic_resonant_with_specified_p_and_x_value, gather_dataset, gather_analytic_resonant
from support.orbit_data import gather_analytic_resonant_with_specified_p_and_x_value, gather_dataset, gather_analytic_resonant
from support.plot import plot_given_boxes, plot_Poincare_2D_with_boxes, plot_Poincare_2D_with_balls, plot_Poincare_2D, plot_Poincare_3D, plot_Poincare_analytic, plot_cross_section_x_vy, plot_cross_section_x_vy_individual

from support.rectangle_calculations import Rect_Poincare_2D_get_boxes


data = gather_dataset(
    plot_second_crossings = True,
    load_lunar                   = False,
    load_prograde_resonant    = False,
    load_prograde_resonant_x1    = False,
    load_retrograde_resonant     = False,
    load_retrograde_resonant_x1  = True,
    load_crash                   = False,
    load_circular                = False,
)

# data = gather_dataset(
#     plot_second_crossings = True,
#     load_lunar                   = True,
#     load_prograde_resonant    = True,
#     load_prograde_resonant_x1    = True,
#     load_retrograde_resonant     = True,
#     load_retrograde_resonant_x1  = True,
#     load_crash                   = True,
#     load_circular                = True,
# )



orbits      = data["orbits"]
crossings   = data["crossings"]
families_id = data["families_str"]
earth_crash_x0 = data["earth_crash_x0"]
earth_crash_vx0 = data["earth_crash_vx0"]
earth_crash_vy0 = data["earth_crash_vy0"]
moon_crash_x0  = data["moon_crash_x0"]
moon_crash_vx0  = data["moon_crash_vx0"]
moon_crash_vy0  = data["moon_crash_vy0"]

# # data_analytic = gather_analytic_resonant(max_p = 9, step_dx = 0.001)
# data_analytic = gather_analytic_resonant(max_p = 8, step_dx = 0.001)

# data_analytic = gather_analytic_resonant(max_p = 20, step_dx = 0.0000001)



# data_analytic = gather_analytic_resonant(max_p = 8, step_dx = 0.001)

# orbits_analytic      = data_analytic["orbits"]
# crossings_analytic   = data_analytic["crossings"]
# families_id_analytic = data_analytic["families_str"]



# print("AAAAAAAAAAAAAAAAAA",orbits_analytic)
# print("AAAAAAAAAAAAAAAAAA",crossings_analytic[0][0].size)
# print("AAAAAAAAAAAAAAAAAA",crossings_analytic)

# crossings_analytic[0][2] = [0]*42
# print("AAAAAAAAAAAAAAAAAA",crossings_analytic[0][2])


# plot_Poincare_2D(orbits, crossings, families_id, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")

# plot_Poincare_2D(orbits_analytic, crossings_analytic, families_id_analytic, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")

# plot_Poincare_3D(orbits, crossings, families_id, 
#                 earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                 plot_second_crossings = True)

# plot_Poincare_3D(orbits_analytic, crossings_analytic, families_id_analytic, 
#                 earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                 plot_second_crossings = True)






#------------------------------------------------------------------------------------------------------------------------------
#data_analytic1 = gather_analytic_resonant(max_p = 8, step_dx = 0.001)
##data_analytic1 = gather_analytic_resonant_with_specified_p_and_x_value(max_p = 8, step_dx = 0.001, x_value = 0.05)
data_analytic1 = gather_analytic_resonant_with_specified_p_and_x_value(max_p = 8, x_value = 0.1)

orbits_analytic1      = data_analytic1["orbits"]
crossings_analytic1   = data_analytic1["crossings"]
families_id_analytic1 = data_analytic1["families_str"]

# print("orbits_analytic (8,0.1):",orbits_analytic1)
# print("crossings_analytic (8,0.05):",crossings_analytic1)
# print("families_id_analytic (8,0.05):",families_id_analytic1)

# print("data_analytic1:",data_analytic1)


data_analytic2 = gather_analytic_resonant_with_specified_p_and_x_value(max_p = 14, x_value = 0.05)

orbits_analytic2      = data_analytic2["orbits"]
crossings_analytic2   = data_analytic2["crossings"]
families_id_analytic2 = data_analytic2["families_str"]

# print("orbits_analytic (9,0.1):",orbits_analytic2)
#print("crossings_analytic (8,0.1):",crossings_analytic2)
# print("families_id_analytic (8,0.1):",families_id_analytic2)

# print("data_analytic2:",data_analytic2)


# plot_Poincare_2D_with_balls(orbits_analytic2, crossings_analytic2, families_id_analytic2, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy", radius=0.02)



# new_merged_orbits = [orbits_analytic1[0],orbits_analytic2[0]]
# print("NEW MERGED ORBITS:", new_merged_orbits)




#####FROM HERE UNTIL "------------------------------------------", THE FOLLOWING IS ALL ABOUT MY merge_datasets FUNCTION TO GET AND PLOT ORBITS WITH TOTALLY DIFFERENT p AND x VALUES.



from typing import Dict, List, Tuple, Sequence
import numpy as np
    
######MOVE THIS TO THE orbit_data.py NOTE TO SELF NOTE TO SELF NOTE TO SELF NOTE TO SELF NOTE TO SELF NOTE TO SELF DO THIS NEXT TIME

def merge_datasets(dataset1: Dict[str, object], dataset2: Dict[str, object]) -> Dict[str, object]:
    """
    Merge the datasets
    """
    
    merged_orbits = dataset1["orbits"] + dataset2["orbits"]
    merged_crossings = dataset1["crossings"] + dataset2["crossings"]
    merged_families_id = dataset1["families_str"] + dataset2["families_str"]
    
    return {
        "orbits":         merged_orbits,
        "crossings":      merged_crossings,
        "earth_crash_x0": np.array([]),   # none created here
        "earth_crash_vx0": np.array([]),
        "earth_crash_vy0": np.array([]),
        "moon_crash_x0":  np.array([]),
        "moon_crash_vx0": np.array([]),
        "moon_crash_vy0": np.array([]),
        "families_str":   merged_families_id,
    }
    # print(merged)
    # print("LBHGYFUTDYCFJVHGKJB", np.concatenate([list1[0], list2[0]]))



merged_data = merge_datasets(data_analytic1, data_analytic2)

# plot_Poincare_2D(merged_data["orbits"], merged_data["crossings"], merged_data["families_str"], 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")

# plot_Poincare_3D(merged_data["orbits"], merged_data["crossings"], merged_data["families_str"],
#                 earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                 plot_second_crossings = True)





######  print("MERGED ORBITS MERGED ORBITS MERGED ORBITS:", merge_lists(orbits_analytic1, orbits_analytic2))


###### merged_orbits = merge_singleton_datasets(data_analytic1["orbits"], data_analytic2["orbits"])
###### print("MERGED ORBITS:", merged_orbits)

def merge_lists(list1, list2):
    return list1 + list2

merged_orbits = merge_lists(orbits_analytic1, orbits_analytic2)
merged_crossings = merge_lists(crossings_analytic1, crossings_analytic2)
merged_families_id = merge_lists(families_id_analytic1, families_id_analytic2)

# plot_Poincare_2D(merged_orbits, merged_crossings, merged_families_id, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")






data_analytic3 = gather_analytic_resonant(max_p = 9, step_dx = 0.1)

orbits_analytic3      = data_analytic3["orbits"]
crossings_analytic3   = data_analytic3["crossings"]
families_id_analytic3 = data_analytic3["families_str"]

###### print("orbits_analytic3:",orbits_analytic3)

#print("data_analytic3:",data_analytic3)


# plot_Poincare_2D(merged_orbits, merged_crossings, merged_families_id, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")


#------------------------------------------------------------------------------------------------------------------------------







#------------------------------------------------------------------------------------------------------------------------------


# plot_Poincare_2D_with_boxes(merged_orbits, merged_crossings, merged_families_id, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy",square_length=0.04)

# # print("JFGHKLGKFDSTHFJHGKJLKJLKGF", len(merged_orbits))


# # for (x_vals, vx_vals, vy_vals, label) in merged_orbits:
# #     print("NJKHBLJVGCJFHGVKBLN",x_vals.size)


data_analytic3 = gather_analytic_resonant(max_p = 9, step_dx = 0.1)

orbits_analytic3      = data_analytic3["orbits"]
crossings_analytic3   = data_analytic3["crossings"]
families_id_analytic3 = data_analytic3["families_str"]


# plot_Poincare_2D_with_boxes(orbits_analytic3, crossings_analytic3, families_id_analytic3, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy",square_length=0.04)




boxes = Rect_Poincare_2D_get_boxes(merged_orbits, merged_crossings, plot_second_crossings = True, which="vy", square_length=0.04)

# boxes = Rect_Poincare_2D_get_boxes(orbits_analytic3, crossings_analytic3, plot_second_crossings = True, which="vy", square_length=0.04)

# print("ABVCASFGHJED")

# for box in boxes:
#         print("BOX IS THE FOLLOWING", box)

plot_given_boxes(boxes, earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, which="vy")

#------------------------------------------------------------------------------------------------------------------------------














# # data_analytic1 = gather_analytic_resonant_with_specified_p_and_x_value(max_p = 8, step_dx = 0.01, x_value = 0.05)
# data_analytic1 = gather_analytic_resonant_with_specified_p_and_x_value(max_p = 8, x_value = 0.05)

# orbits_analytic1      = data_analytic1["orbits"]
# crossings_analytic1   = data_analytic1["crossings"]
# families_id_analytic1 = data_analytic1["families_str"]

# # print("orbits_analytic (8,0.01,0.05):",orbits_analytic1)

# plot_Poincare_2D(orbits_analytic1, crossings_analytic1, families_id_analytic1, 
#                   earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                   plot_second_crossings = True, which="vy")

# # plot_Poincare_2D_with_balls(orbits_analytic1, crossings_analytic1, families_id_analytic1, 
# #                   earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
# #                   plot_second_crossings = True, which="vy", radius=0.02)





###----------------------------------

# data_analytic1 = gather_analytic_resonant(max_p = 8, step_dx = 0.01)

# orbits_analytic1      = data_analytic1["orbits"]
# crossings_analytic1   = data_analytic1["crossings"]
# families_id_analytic1 = data_analytic1["families_str"]


## print("orbits_analytic (8,0.01,0.05):",orbits_analytic1)

# plot_Poincare_2D(orbits_analytic1, crossings_analytic1, families_id_analytic1, 
#                   earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                   plot_second_crossings = True, which="vy")

# plot_Poincare_2D_with_balls(orbits_analytic1, crossings_analytic1, families_id_analytic1, 
#                   earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                   plot_second_crossings = True, which="vy", radius=0.01)

# plot_Poincare_3D(orbits_analytic1, crossings_analytic1, families_id_analytic1, 
#                 earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                 plot_second_crossings = True)


####-----------------------------------







# data_analytic = gather_analytic_resonant(max_p = 12, step_dx = 0.001)

# orbits_analytic      = data_analytic["orbits"]
# crossings_analytic   = data_analytic["crossings"]
# families_id_analytic = data_analytic["families_str"]

# plot_Poincare_2D(orbits_analytic, crossings_analytic, families_id_analytic, 
#                  earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0, 
#                  plot_second_crossings = True, which="vy")








# # 4) Analytic resonance overlay (catalog orbits optional): p:1 families over accessible region
# #    Use to visualize multiple p:1 branches (set res_num) and optionally overlay family clouds.
# plot_Poincare_analytic(
#     orbits, crossings,
#     res_num=15,           # number of p:1 curves to show (p = 2..res_num-1)
#     show_families=False,  # if True, scatter loaded orbits on top
#     retro=True,           # plot retrograde (+/- sign convention inside)
# )



# plot_cross_section_x_vy(orbits_analytic, crossings_analytic, families_id_analytic,
#                        earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
#                        vx_min = -5, vx_max = 5, num_intervals=10)

# plot_cross_section_x_vy(orbits, crossings, families_id, 
#                        earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
#                        vx_min = -5, vx_max = 5, num_intervals=10)

# plot_cross_section_x_vy_individual(orbits_analytic, crossings_analytic, families_id_analytic,
#                                    earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
#                                    vx_min = -5, vx_max = 5, num_intervals=10, save_fig=True)

# plot_cross_section_x_vy_individual(orbits, crossings, families_id,
#                                    earth_crash_x0, earth_crash_vx0, earth_crash_vy0, moon_crash_x0, moon_crash_vx0, moon_crash_vy0,
#                                    vx_min = -5, vx_max = 5, num_intervals=10, save_fig=True)



