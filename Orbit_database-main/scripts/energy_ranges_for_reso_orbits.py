"""
CR3BP Resonance Interval Plots

This script visualizes intervals where certain Earth–Moon CR3BP resonant orbits exist.
For each resonance label (e.g., "m:n"), it plots:
  • the x0 ranges as horizontal line segments, and
  • the Jacobi constant (J) ranges as horizontal line segments
    (with a reference line at J = 3.172 for the E2 energy).

Toggle `retro` to switch between prograde (retro=False) and retrograde (retro=True) datasets.
Outputs:
  • Figures saved under: {BASE_PATH}/Figures/energy_ranges_resos/
    - x0_ranges_resonant_retro_{retro}.png
    - Energy_ranges_resonant_retro_{retro}.png
"""

import matplotlib.pyplot as plt
import numpy as np
from support.constants import BASE_PATH

# ============================================
# 1) Define your resonance data (multi-ranges)
#    Each row: ( resonance_label,
#                list_of_x0_intervals,
#                list_of_J_intervals )
# ============================================
retro = False
if retro:
    data = [
        ("2:1", [(0.032, 0.972)], [(0.048, 1.494)]),
        ("3:1", [(0.001, 0.595)], [(0.098, 1.836)]),
        ("4:1", [(0.395, 0.769)], [(1.286, 2.301)]),
        ("5:1", [(0.341, 0.622)], [(1.774, 2.352)]),
        ("5:2", [(0.014, 0.682)], [(-0.213, 1.6)]),
        ("6:1", [(0.303, 0.582)], [(2.216, 3.039)]),
        ("7:1", [(0.273, 0.438)], [(2.624, 2.875)]),
        ("7:2", [(0.43, 0.595)], [(0.102, 1.073)]),
        ("7:3", [(0.027, 0.925)], [(-0.514, 1.475)]),
        ("8:1", [(0.25, 0.477)], [(3.007, 3.738)]),
        ("8:3", [(0.005, 0.249)], [(0.552, 1.723)]),
        ("9:1", [(0.016, 0.251)], [(3.007, 7.087)]),
        ("9:2", [(0.395, 0.402)], [(1.286, 1.287)]),
        ("9:4", [(0.03, 0.249)], [(0.518, 1.449)]),
        ("10:1", [], []),  # No solutions available
        ("10:3", [(0.403, 0.595)], [(0.098, 1.12)])
    ]
else:
    data = [
        ("2:1",   [(0.0717, 0.5038)], [(2.393, 3.172)]),
        ("3:1",   [(0.0022, 0.3004)], [(2.495, 3.37)]),
        ("4:1",   [(0.1005, 0.3790)], [(3.400, 3.773)]),
        ("5:1",   [(0.3420, 0.6420)], [(3.200, 4.083)]),
        ("5:2",   [(0.1750, 0.8100)], [(3.075,3.31)]),
        ("6:1",   [(0.3020, 0.5800)], [(3.629, 4.389)]),
        ("7:1",   [(0.2730, 0.5250)], [(3.916, 4.688)]),
        ("7:2",   [(0.4340, 0.8390)], [(2.726, 3.616)]),
        ("7:3",   [(0.2100, 0.7920)], [(2.967, 3.26)]),
        ("8:1",   [(0.2490, 0.47), (0.474, 0.4790)], [(4.24,4.981)]), 
        ("8:3",   [(0.0110, 0.9410)], [(2.382, 3.36)]),
        ("9:1",   [(0.083,0.441)],[(4.565,5.267)]),
        ("9:2",   [(0.3670, 0.7050)], [(3.142, 3.928)]),
        ("9:4",   [(0.05, 0.85)], [(2.409, 3.233)]),
        ("10:1",  [(0.014, 0.408)], [(4.9, 5.546)]),
    ]


# -------------------------------------------------
# 2) Helper: parse m:n ratio for sorting by m/n
# -------------------------------------------------
def parse_ratio(label):
    """Parse a string like '7:3' and return float(7)/float(3)."""
    m, n = label.split(':')
    return float(m) / float(n)

# Build (label, ratio, x0_intervals, J_intervals), falling back to inf if parsing fails.
data_with_ratio = []
for lbl, x0_ints, j_ints in data:
    try:
        ratio = parse_ratio(lbl)
    except Exception:
        ratio = np.inf
    data_with_ratio.append((lbl, ratio, x0_ints, j_ints))

# Sort by numeric ratio (ascending)
data_sorted = sorted(data_with_ratio, key=lambda t: t[1])

# Unpack sorted columns
resonances = [t[0] for t in data_sorted]
ratios     = [t[1] for t in data_sorted]
x0_ranges  = [t[2] for t in data_sorted]
J_ranges   = [t[3] for t in data_sorted]

# Assign a y-position row per resonance
y_positions = np.arange(len(resonances))

# -------------------------------------------------
# 3) Plot x0 ranges (supports multiple sub-ranges)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

for i, intervals in enumerate(x0_ranges):
    for xmin, xmax in intervals:
        ax.hlines(y_positions[i], xmin, xmax, color='black', linewidth=2)
        ax.plot([xmin, xmax], [y_positions[i], y_positions[i]], 'o', color='black', markersize=3)

ax.set_yticks(y_positions)
ax.set_yticklabels(resonances)
ax.invert_yaxis()  # Place lower m:n (e.g., 2:1) near the top (optional)
ax.set_xlabel(r"$x_0$ range")
ax.set_title("CR3BP Resonances: $x_0$ Ranges")

plt.tight_layout()
base_path = BASE_PATH
plt.savefig(
    f"{base_path}/Figures/energy_ranges_resos/x0_ranges_resonant_retro_{retro}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# -------------------------------------------------
# 4) Plot J ranges (supports multiple sub-ranges)
#    Includes vertical reference line at J = 3.172 (E2 energy).
# -------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 6))

for i, intervals in enumerate(J_ranges):
    for jmin, jmax in intervals:
        ax2.hlines(y_positions[i], jmin, jmax, color='black', linewidth=2)
        ax2.plot([jmin, jmax], [y_positions[i], y_positions[i]], 'o', color='black', markersize=3)

ax2.axvline(x=3.172, color='black', linestyle='--', label='E2 energy')

ax2.set_yticks(y_positions)
ax2.set_yticklabels(resonances)
ax2.invert_yaxis()  # Optional
ax2.set_xlabel("Jacobi constant (J)")
ax2.set_title("CR3BP Resonances: Jacobi Ranges")
ax2.legend(loc='best')

plt.tight_layout()
plt.savefig(
    f"{base_path}/Figures/energy_ranges_resos/Energy_ranges_resonant_retro_{retro}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()