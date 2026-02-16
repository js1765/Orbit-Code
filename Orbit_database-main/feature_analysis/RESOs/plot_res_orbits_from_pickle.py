#!/usr/bin/env python3
"""
plot_resonant_orbits.py

This script reads resonant orbit data from a pickle file and plots the full orbit
trajectories in the x–y plane. You can choose to plot either 5:1 or 6:1 resonant orbits,
and you can specify a subsampling step (i.e. plot every nth orbit).

Usage:
    python plot_resonant_orbits.py --resonance 6:1 --step 2
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as pe


def load_orbits(resonance):
    """
    Load resonant orbits from the appropriate pickle file.
    
    Parameters:
        resonance (str): The resonance type. Should be "5:1" or "6:1".
    
    Returns:
        list: A list of orbit dictionaries.
    """
    # Define the expected pickle file names (adjust path as needed)
    if resonance == "5:1":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_5_1_database.pkl"
    elif resonance == "6:1":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_6_1_database.pkl"
    elif resonance == "7:1":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_7_1_database.pkl"
    elif resonance == "10:1":
        pickle_file = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/data/resonant_orbits_10_1_database.pkl"
    else:
        raise ValueError("Invalid resonance specified. Choose either '5:1' or '6:1'.")

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file '{pickle_file}' not found.")

    with open(pickle_file, "rb") as f:
        orbits = pickle.load(f)
    print(f"Loaded {len(orbits)} orbits from '{pickle_file}'.")
    return orbits

def plot_orbits(orbit_trajectories):
    """
    Plot orbit trajectories in the x-y plane on a black background,
    using a neon-like colormap. Axes and labels are removed for a
    purely aesthetic, 'art of science' style visualization.
    """
    plt.figure(figsize=(10, 8), facecolor='black')
    ax = plt.gca()
    #ax.set_facecolor('black')
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_frame_on(False)

    # Gather initial x for coloring
    x0_list = []
    for orbit in orbit_trajectories:
        if "initial_state" in orbit:
            x0 = orbit["initial_state"][0]
        else:
            x0 = orbit["states"][0, 1]
        x0_list.append(x0)
    x0_array = np.array(x0_list)

    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=x0_array.min(), vmax=x0_array.max())

    for orbit in orbit_trajectories:
        if "initial_state" in orbit:
            x0 = orbit["initial_state"][0]
        else:
            x0 = orbit["states"][0, 1]
        color = cmap(norm(x0))
        
        x = orbit["states"][:, 0]
        y = orbit["states"][:, 1]

        # Plot the orbit line
        line, = plt.plot(x, y, color=color, linewidth=1)

        # Add a few path effects to simulate a glow:
        # - Multiple strokes with increasing size, decreasing alpha
        # - Then the normal line on top
        #line.set_path_effects([
        #    pe.Stroke(linewidth=3.5, foreground=color, alpha=0.2),  # Subtle outer glow
        #    pe.Stroke(linewidth=2.5, foreground=color, alpha=0.5),  # Slightly tighter glow
        #    pe.Normal()  # Original line (linewidth=2)
        #])

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot resonant orbits in the x-y plane.")
    parser.add_argument(
        "--resonance",
        type=str,
        choices=["5:1", "6:1", "7:1", "8:1","10:1"],
        default="10:1",
        help="Which resonant orbits to plot (5:1 or 6:1). Default: 6:1"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Plot every nth orbit. For example, --step 2 will plot every second orbit. Default: 1"
    )
    args = parser.parse_args()

    orbits = load_orbits(args.resonance)
    plot_orbits(orbits[::args.step])

if __name__ == "__main__":
    main()