#####IGNORE THIS FILE. 
# I was just using it to test some stuff while I tried to switch Jannik's original code to use relative paths/work on my computer.


from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
print(base_dir)

# filename = base_dir / "data" / "circular_orbits_moon_database.pkl"
# print(filename)