# from __future__ import annotations
# from typing import List, Tuple, Union

from pydoc import Helper
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



# from matplotlib.axes import Axes
# from support.constants import cr3bp, mu1, mu2, R_earth, R_moon, earth_collision_radius, E0, jacobimin, L2_info, L1_info, U_tilde, BASE_PATH  # type: ignore
# from support.helpers import earth_crash_vy_branches, generate_x_ranges, crash_surface_vy






# put the parent directory (the one containing "support") on Python’s search path
sys.path.append(str(Path(__file__).resolve().parent.parent))



# # def get_total_rect_area(rects)





# HAS_BODIES = all(name in globals() for name in ("mu1", "mu2", "R_earth", "R_moon"))


# # ------------------------------------------------------------------
# #  Internal helper – draw discs
# # ------------------------------------------------------------------

# def _add_bodies(ax):
#     if not HAS_BODIES:
#         return
#     ax.scatter([-mu2], [0], s=200, c="tab:blue", alpha=0.3) #These sizes (the 's' values) are arbitrary to make them visible, they are not to scale or anything!
#     ax.scatter([ mu1], [0], s=50,  c="tab:gray", alpha=0.3)
    
    
    


# def calculate_mega_box(x_vals, y_vals, square_length):
#        # if len(x_vals) == 0:
#        #        return None  # No points, no box
#        # x_min = np.min(x_vals) - square_length/2
#        # x_max = np.max(x_vals) + square_length/2
#        # y_min = np.min(y_vals) - square_length/2
#        # y_max = np.max(y_vals) + square_length/2
#        # return (x_min, x_max, y_min, y_max)
       
#        if len(x_vals) == 0:
#               return None  # No points, no box
       
       
       
       

# def split_rect(x_coords, rects): #x_coords is the x value for the left corners of the box (so if the box is centred at (x,y), then its x_coord is x-l/2, where l is the (horizontal) length of the box.
       

       
       




#------------------------------------------------------------------------------
def Rect_Poincare_2D_get_boxes(all_orbits, all_crossings, plot_second_crossings=True, which="vy", square_length=0.01):
    
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111)
#     _add_bodies(ax)
#     x_min = -0.8
#     x_max = L2_info[0][0]

    def select_velocity(vx_vals, vy_vals):
        return vy_vals if which == "vy" else vx_vals

    boxes = []

    ### Creates the box object
    def add_box(x_vals, y_vals, square_length):
        for x, y in zip(x_vals, y_vals):
            l = square_length/2
            rect = Rectangle((x - l/2, y-l/2), l, l)
            boxes.append(rect)
            
            

    ### Adds a boxes around each given point in the first crossings
    for (x_vals, vx_vals, vy_vals, label) in all_orbits:
        velocity_data = select_velocity(vx_vals, vy_vals)
        if x_vals.size > 0:
            add_box(x_vals, velocity_data, square_length)

            
            
    ### Adds a boxes around each given point in the second crossings (if desired)
    if plot_second_crossings:
        for (x_vals, vx_vals, vy_vals, label) in all_crossings:
            velocity_data = select_velocity(vx_vals, vy_vals)
            add_box(x_vals, velocity_data, square_length)

    
#     for box in boxes:
#         ax.add_patch(box)

#     plt.show()
    
    return boxes

#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
def calculate_mega_box(x_vals, y_vals, square_length):
       # if len(x_vals) == 0:
       #        return None  # No points, no box
       # x_min = np.min(x_vals) - square_length/2
       # x_max = np.max(x_vals) + square_length/2
       # y_min = np.min(y_vals) - square_length/2
       # y_max = np.max(y_vals) + square_length/2
       # return (x_min, x_max, y_min, y_max)
       
       if len(x_vals) == 0:
              return None  # No points, no box


#------------------------------------------------------------------------------











