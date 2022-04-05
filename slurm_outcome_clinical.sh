#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=hn_clinical   # sensible name for the job
#SBATCH --mem=16G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/fcn-%A.out
#SBATCH --error=outputs/fcn-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 3 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

if [ ! -d "$TMPDIR/$USER/hn_delin" ]
    then
    echo "Didn't find dataset folder. Copying files..."
    mkdir --parents $TMPDIR/$USER/hn_delin
    fi

for f in $(ls $HOME/datasets/headneck/*)
    do
    FILENAME=`echo $f | awk -F/ '{print $NF}'`
    echo $FILENAME
    if [ ! -f "$TMPDIR/$USER/hn_delin/$FILENAME" ]
        then
        echo "copying $f"
        cp -r $HOME/datasets/headneck/$FILENAME $TMPDIR/$USER/hn_delin/
        fi
    done


echo "Finished seting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
# export ITER_PER_EPOCH=200
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
singularity exec --nv deoxys-mar22-polished.sif python experiment_outcome.py $1 /net/fs-1/Ngoc/hnperf/$2_run1 --temp_folder $SCRATCH/hnperf/$2_run1 --analysis_folder $SCRATCH/analysis/$2_run1 --epochs $3 ${@:4}
singularity exec --nv deoxys-mar22-polished.sif python experiment_outcome.py $1 /net/fs-1/Ngoc/hnperf/$2_run2 --temp_folder $SCRATCH/hnperf/$2_run2 --analysis_folder $SCRATCH/analysis/$2_run2 --epochs $3 ${@:4}
singularity exec --nv deoxys-mar22-polished.sif python experiment_outcome.py $1 /net/fs-1/Ngoc/hnperf/$2_run3 --temp_folder $SCRATCH/hnperf/$2_run3 --analysis_folder $SCRATCH/analysis/$2_run3 --epochs $3 ${@:4}
singularity exec --nv deoxys-mar22-polished.sif python experiment_outcome.py $1 /net/fs-1/Ngoc/hnperf/$2_run4 --temp_folder $SCRATCH/hnperf/$2_run4 --analysis_folder $SCRATCH/analysis/$2_run4 --epochs $3 ${@:4}
singularity exec --nv deoxys-mar22-polished.sif python experiment_outcome.py $1 /net/fs-1/Ngoc/hnperf/$2_run5 --temp_folder $SCRATCH/hnperf/$2_run5 --analysis_folder $SCRATCH/analysis/$2_run5 --epochs $3 ${@:4}

singularity exec --nv deoxys-mar22-polished.sif python ensemble_outcome.py /net/fs-1/Ngoc/hnperf/$2_ run1,run2,run3,run4,run5 --merge_name all

# echo "Finished training. Post-processing results"

# singularity exec --nv deoxys.sif python -u post_processing.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}

# echo "Finished post-precessing. Running test on best model"

# singularity exec --nv deoxys.sif python -u run_test.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}
