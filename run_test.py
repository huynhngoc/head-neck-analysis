"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import Experiment
# from deoxys.utils import read_file
import argparse
import os
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
import customize_obj

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder",
                        default='', type=str)
    parser.add_argument("--monitor", default='', type=str)

    args, unknown = parser.parse_known_args()

    # 42 images for 2d, 15 images for 3d
    img_num = 42

    if '3d' in args.log_folder:
        img_num = 40

    best_model = customize_obj.PostProcessor(
        args.log_folder,
        temp_base_path=args.temp_folder).best_model

    (
        customize_obj.AnalysisExperiment(
            log_base_path=args.log_folder,
            temp_base_path=args.temp_folder)
        .from_file(best_model)
        .run_test(masked_images=[i for i in range(img_num)])
    )

    if '2d' in args.log_folder:
        customize_obj.PostProcessor(
            args.log_folder,
            temp_base_path=args.temp_folder,
            map_meta_data=args.meta,
            run_test=True
        ).map_2d_meta_data().calculate_fscore_single().merge_2d_slice(
        ).calculate_fscore()
