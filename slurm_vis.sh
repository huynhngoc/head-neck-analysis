#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=hn_vis   # sensible name for the job
#SBATCH --mem=16G                 # Default memory per CPU is 3GB.
#SBATCH --partition=smallmem # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/vis-%A.out
#SBATCH --error=outputs/vis-%A.out

# If you would like to use more please adjust this.


# Run experiment
singularity exec --nv deoxys.sif python experiment_vis.py $HOME/hnperf/$1
