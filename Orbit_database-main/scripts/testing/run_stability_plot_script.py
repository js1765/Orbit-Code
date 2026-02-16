import subprocess

# List of resonance values to test
resonances = [
    "2.0:1.0",
    "3.0:1.0",
    "4.0:1.0",
    "5.0:1.0",
    "5.0:2.0",
    "6.0:1.0",
    "7.0:1.0",
    "7.0:2.0",
    "7.0:3.0",
    "8.0:1.0",
    "8.0:3.0",
    "9.0:1.0",
    "9.0:2.0",
    "9.0:4.0",
    "10.0:1.0",
    "10.0:3.0"
]

# Path to the Python script to be executed
script_path = "/Users/jannik/Documents/PhD_Princeton/Research/SSA/orbit_database/feature_analysis/RESOs/plot_reso_stability_from_pickle_or_url.py"

# Loop over each resonance value and run the script with the corresponding argument
for res in resonances:
    print(f"Running for resonance: {res}")
    subprocess.run(["python", script_path, "--resonance", res])