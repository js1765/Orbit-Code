#!/bin/bash
#SBATCH --job-name=4-1      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1    # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=jg3607@princeton.edu

module purge

module load anaconda3/2022.5
conda activate orbit_generation
module load boost/1.73.0 

python /home/jg3607/SSA_project/Orbit_database/feature_analysis/RESOs/create_reso_database_robust.py --p 4 --q 1 --start 0.398 --stop 0.774 2>/dev/null