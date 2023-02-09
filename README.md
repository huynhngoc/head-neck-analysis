# Head and Neck cancer analysis

Start by running `setup.sh` to download the singularity container
Then, submit slurm jobs like this:

```bash
sbatch slurm.sh config/2d_unet.json 2d_unet 200
```

Which will load the setup from the `config/2d_unet.json` file, train for 200 epochs
and store the results in the folder `$HOME/logs/hn_perf/2d_unet/`.

To customize model and prediction checkpoints

```
sbatch slurm.sh config/3d_vnet_32_normalize.json 3d_vnet_32_normalize 100 --model_checkpoint_period 5 --prediction_checkpoint_period 5

```

To continue an experiment
```
sbatch slurm_cont.sh model.030.h5 3d_vnet_32_normalize 100 --model_checkpoint_period 5 --prediction_checkpoint_period 5
```

To plot performance
```
sbatch slurm_vis.sh 3d_vnet_32_normalize
```

To run test
```
sbatch slurm_test.sh 3d_vnet_32
```
or
```
sbatch slurm_test.sh 3d_vnet_32 --best_epoch <BEST_EPOCH_NUMBER>
```

External validation
```
sbatch slurm_external.sh ds_config.json path/to/log/folder
```


To transfer learning to new dataset
```
sbatch slurm_transfer.sh ds_config.json path/to/log/folder <epoch_num> --initial_epoch <init_epoch>
```


Alternatively, if your cluster does not have slurm installed, simply omit the `sbatch`
part of the call above, thus running

```bash
./slurm.sh config/2d_unet.json 2d_unet 200
```

Manually build
```
singularity build --fakeroot deoxys.sif Singularity
```

Remember to login to a gpu session to use the gpu
```
qlogin --partition=gpu --gres=gpu:1
```
