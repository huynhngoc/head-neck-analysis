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
if [ $# -lt 4 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

if [ ! -d "$TMPDIR/$USER/hn_delin" ]
    then
    echo "Didn't find dataset folder. Copying files..."
    mkdir --parents $TMPDIR/$USER/hn_delin
    fi

for f in $(ls $PROJECTS/ngoc/datasets/headneck/*)
    do
    FILENAME=`echo $f | awk -F/ '{print $NF}'`
    echo $FILENAME
    if [ ! -f "$TMPDIR/$USER/hn_delin/$FILENAME" ]
        then
        echo "copying $f"
        cp -r $PROJECTS/ngoc/datasets/headneck/$FILENAME $TMPDIR/$USER/hn_delin/
        fi
    done


echo "Finished seting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
# export ITER_PER_EPOCH=200
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray

fold_list=$3
for folds in ${fold_list//,/ }
do
    singularity exec --nv deoxys-2023-feb-fixed.sif python experiment_outcome_clinical.py $1_$fold.json $PROJECTS/ngoc/hnperf/$2_$fold --temp_folder $SCRATCH_PROJECTS/ceheads/hnperf/$2_$fold --analysis_folder $SCRATCH_PROJECTS/ceheads/analysis/$2_$fold --epochs $4 ${@:5}
done

test_fold_idx=${fold: -1}

singularity exec --nv deoxys-2023-feb-fixed.sif python -u ensemble_outcome.py $PROJECTS/ngoc/hnperf/$1_ $3 --merge_name test_fold$test_fold_idx

for fold in ${fold_list//,/ }
do
    singularity exec --nv deoxys-2023-feb-fixed.sif python outcome_external_clinical.py external_config/outcome_maastro_tabular.json $PROJECTS/ngoc/hnperf/$2_$fold --temp_folder $SCRATCH_PROJECTS/ceheads/hnperf/$2_$fold --analysis_folder $SCRATCH_PROJECTS/ceheads/analysis/$2_$fold ${@:5}
done

fold_maastro=""
for fold in ${fold_list//,/ }
do
    echo $fold
    fold_maastro+=",$fold"
    fold_maastro+="_outcome_maastro_tabular"
done
fold_maastro=${fold_maastro:1}
echo $fold_maastro

singularity exec --nv deoxys-2023-feb-fixed.sif python -u ensemble_outcome.py $PROJECTS/ngoc/hnperf/$1_ $fold_maastro --merge_name maastro_fold$test_fold_idx
