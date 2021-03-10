"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import Experiment, ExperimentPipeline
# from deoxys.utils import read_file
import argparse
import os
import shutil
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
import customize_obj

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--best_epoch", default=5, type=int)
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--meta", default='patient_idx,slice_idx', type=str)
    parser.add_argument("--monitor", default='', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    if 'patch' in args.log_folder:
        analysis_folder = args.analysis_folder
    else:
        analysis_folder = ''

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    # copy to another location
    log_folder = args.log_folder + '_' + args.dataset_file[:-5].split('/')[-1]
    if not os.path.exists(log_folder):
        shutil.copytree(args.log_folder, log_folder)

    ex = ExperimentPipeline(
        log_base_path=log_folder,
        temp_base_path=args.temp_folder
    )
    try:
        ex = ex.load_best_model(
            recipe='auto',
            analysis_base_path=analysis_folder,
            map_meta_data=meta,
        )
    except Exception as e:
        print(e)
        ex.from_file(log_folder +
                     f'/model/model.{args.best_epoch:03d}.h5')
    ex.run_external(
        args.dataset_file
    ).apply_post_processors(
        recipe='auto',
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
        run_test=True
    ).plot_3d_test_images(best_num=2, worst_num=2)
