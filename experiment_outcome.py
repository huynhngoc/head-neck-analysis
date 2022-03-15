"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

import customize_obj
# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
# import os
# import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='AUC', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)
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

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print('training from configuration', args.config_file,
          'and saving log files to', args.log_folder)
    print('Unprocesssed prediction are saved to', args.temp_folder)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_full_config(
        args.config_file
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
    ).apply_post_processors(
        map_meta_data=meta,
        metrics=['AUC', 'roc_auc', 'f1', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta', 'matthews_corrcoef'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'tf', 'sklearn'],
        process_functions=[None, None, binarize, None, None, None, binarize],
        metrics_kwargs=[{}, {}, {}, {}, {}, {}, {'metric_name': 'mcc'}]
    ).plot_performance().load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode
    ).run_test().apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['AUC', 'roc_auc', 'f1', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta', 'matthews_corrcoef'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'tf', 'sklearn'],
        process_functions=[None, None, binarize, None, None, None, binarize],
        metrics_kwargs=[{}, {}, {}, {}, {}, {}, {'metric_name': 'mcc'}]
    )
