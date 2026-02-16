import numpy as np
from pathlib import Path
import sys
# put the parent directory (the one containing "support") on Python’s search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from support.helpers import propagate_orbit
# from support.plot import plot_2D_trajectory
from support.plot import plot_2D_trajectory, plot_2D_trajectory_earth_and_moon

# from support.constants import BASE_PATH, jacobimin_JPL, jacobimin, mu1, mu2, E0, U_tilde, earth_collision_radius, L2_info
# print(E0)
# print()



# initial_cond = np.array([1.0773030697504022,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        -0.46978550159437038,  0.00000000e+00])
# period = 1.2766784653764811
######[t,x,y,z,vx,vy,vz]
# initial_cond = np.array([1.155695000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 22.0


# initial_cond = np.array([0.000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  1000.00000000e+00])
# period = 10.0


# initial_cond = np.array([10.000000e+00,  0.00000000e+00,  0.00000000e+00,  -5.00000000e+00,
#        1.2766784653764811])

# initial_cond = np.array([1.000000e+00,  1.00000000e+00,  0.00000000e+00,  5.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 10.0


# initial_cond = np.array([-100.000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 10.0

# initial_cond = np.array([1.100000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 5.0

###COOL ORBIT AROUND THE MOON
# initial_cond = np.array([1.030000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 5.0

# ###COOL ORBIT AROUND THE EARTH
# initial_cond = np.array([0.50000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 5.0

###COOL ORBIT AROUND THE MOON
# initial_cond = np.array([0.850000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0

###ORBIT AROUND THE EARTH WITH A SHARP(ER) POINT NEAR THE MOON
# initial_cond = np.array([0.83680000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.000000000e+00])
# period = 20.0

# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory(trajectory,moon_plot=False)



# #### IN THESE NEXT TWO, NOTICE THAT IT CHANGES BETWEEN 0.836899983 AND 0.836899984
# initial_cond = np.array([0.8368999830000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory(trajectory,moon_plot=False)
# initial_cond = np.array([.8368999840000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory(trajectory,moon_plot=False)





###------------------------------------------------------------------------------------------------------------------------------


# #### IN THESE NEXT TWO, NOTICE THAT IT CHANGES BETWEEN 0.836899983 AND 0.836899984
# initial_cond = np.array([0.8368999830000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)
# initial_cond = np.array([0.8368999840000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)

### NOTE THAT the Earth is located at earth_x = -mu2 = 0.012153663111117432.
### Wikipedia says that the earth and moon are 0.3844×10^9 m apart. And the L1 point is 0.32639×10^9 m from the earth to the moon.
### Thus, the L1 point is about 0.84908949011 of the distance from the earth to the moon.
### As our earth is located at -mu2 = 0.012153663111117432 for some reason (not sure why Jannik didnt jsut put it at the origin, but I assume he has his reasons),
### we get that this weird bifurcation point (did I use that term correctly?) is at x = about 0.836899984 + (-mu2) = ###we get that this weird bifurcation point 
### (did I use that term correctly?) is at x = about 0.836899984 + (-mu2) = 0.8490536471111174 + 0.012153663111117432 = 0.8490536471111174
### So, this point is about 0.8490536471111174 of the distance from the earth to the moon, which is very, VERY close to what wikipedia predicts.
### This strange switching behaviour seems to happen right around the L1 point!!!
# print(0.012153663111117432+0.836899984)


###------------------------------------------------------------------------------------------------------------------------------


### Let's see what happens if we do something similar for the L2 point:
### Wikipedia tells us that the L2 point is about 1.16779396462 of the distance from the earth to the moon.
### So, in our case, because Earth is at -mu2 = 0.012153663111117432, we'd expect it to be at about 1.16779396462+0.012153663111117432 = 1.17994762773
### Let's see:
#### IN THESE NEXT TWO, NOTICE THAT IT CHANGES BETWEEN 1.175 AND 1.185
initial_cond = np.array([1.175,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00])
period = 20.0
trajectory = propagate_orbit(initial_cond, period)
plot_2D_trajectory_earth_and_moon(trajectory)
initial_cond = np.array([1.185,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00])
period = 20.0
trajectory = propagate_orbit(initial_cond, period)
plot_2D_trajectory_earth_and_moon(trajectory)

###Wait it doesn't actually seem to really be making a difference.
###But maybe that make sense because now the satellite isnt inbetween two things? So like, it wouldnt fall to one or the other?


###------------------------------------------------------------------------------------------------------------------------------



# ###REALLY COOL ORBIT AROUND THE EARTH
# initial_cond = np.array([0.012153663111117432,  0.500000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)


# ###ANOTHER REALLY COOL ORBIT AROUND THE EARTH, LESS SPIKY, WAY MORE ROUND
# initial_cond = np.array([0.012153663111117432,  0.500000000e+00,  0.00000000e+00,  1.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 20.0
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)



# initial_cond = np.array([0.012153663111117432,  0.300000000e+00,  0.00000000e+00,  5.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 0.25
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)



# ####CAN THIS BE RIGHT???? WHY IS IT GOING **AWAY** FROM THE EARTH/MOON SYSTEM??? 
# #### DOES THIS HAVE TO DO WITH THE ROTATING FRAME??? OR IS IT JUST NONSENSE???
# initial_cond = np.array([0.9878463368888826,  0.300000000e+00,  0.00000000e+00,  -5.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 1
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)




# ####REALLY WEIRD BEHAVIOUR IN THIS ONE, CHECK IT OUT!
# initial_cond = np.array([0.9878463368888826,  0.300000000e+00,  0.00000000e+00,  0.00000000e+00,
#        0.00000000e+00,  0.00000000e+00])
# period = 50
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)



# ###THIS ONE TOO! IT LIKE MAKES A THREE LEAF CLOVER AROUND THE EARTH, AND THEN GOES OUT IN A SPIRAL EVENTUALLY
# ## IT LOOKS LIKE IT SHOULD ACTUALLY BE CRASHING THOUGH...
# initial_cond = np.array([0.9878463368888826,  0.300000000e+00,  0.00000000e+00,  0.00000000e+00,
#        -1.00000000e+00,  0.00000000e+00])
# # period = 20
# # period = 2
# period = 10
# trajectory = propagate_orbit(initial_cond, period)
# plot_2D_trajectory_earth_and_moon(trajectory)






# trajectory = propagate_orbit(initial_cond, period)
# # print(trajectory.size)
# plot_2D_trajectory(trajectory,moon_plot=False)
