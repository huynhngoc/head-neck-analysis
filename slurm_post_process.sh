#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=hn_pp   # sensible name for the job
#SBATCH --mem=16G                 # Default memory per CPU is 3GB.
#SBATCH --partition=smallmem # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/unet-test-%A.out
#SBATCH --error=outputs/unet-test-%A.out

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
singularity exec --nv deoxys.sif python post_processing.py $1 /net/fs-1/Ngoc/hnperf/$2
