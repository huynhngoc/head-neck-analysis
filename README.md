# Head and Neck cancer analysis

Start by running `setup.sh` to download the singularity container
Then, submit slurm jobs like this:

```bash
sbatch slurm.sh config/2d_unet.json 2d_unet 200
```

Which will load the setup from the `config/2d_unet.json` file, train for 200 epochs
and store the results in the folder `$HOME/logs/hn_perf/2d_unet/`.

Alternatively, if your cluster does not have slurm installed, simply omit the `sbatch`
part of the call above, thus running

```bash
./slurm.sh config/2d_unet.json 2d_unet 200
```

Manually build
```
singularity build --fakeroot Singularity deoxys.sif
```

Remember to login to a gpu session to use the gpu
```
qlogin --partition=gpu --gres=gpu:1
```
