#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=uncertainty   # sensible name for the job
#SBATCH --mem=64G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/uncertainty-%A-%a.out
#SBATCH --error=outputs/uncertainty-%A-%a.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 2 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

echo "Finished seting up files."


# Run experiment
# export ITER_PER_EPOCH=200
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
export MAX_SAVE_STEP_GB=0
rm -rf $TMPDIR/ray/*
singularity exec --nv deoxys-2023-feb-fixed.sif python -u run_uncertainty.py $1 $PROJECTS/ngoc/segmentation --dropout_rate $2 --iter $SLURM_ARRAY_TASK_ID

# echo "Finished training. Post-processing results"

# singularity exec --nv deoxys.sif python -u post_processing.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}

# echo "Finished post-precessing. Running test on best model"

# singularity exec --nv deoxys.sif python -u run_test.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}
