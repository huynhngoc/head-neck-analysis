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

    parser = argparse.ArgumentParser()
    parser.add_argument("log_base")
    parser.add_argument("name_list")
    parser.add_argument("--merge_name", default='merge', type=str)
    parser.add_argument("--mode", default='ensemble', type=str)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='AUC', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)

    args, unknown = parser.parse_known_args()

    log_path_list = [args.log_base +
                     name for name in args.name_list.split(',')]
    log_base_path = args.log_base + args.merge_name

    if args.mode == 'ensemble':
        print('Ensemble test results from this list', log_path_list)
    else:
        print('Concatenate test results from this list', log_path_list)

    print('Merged results are save to', log_base_path)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    pp = customize_obj.EnsemblePostProcessor(
        log_base_path=log_base_path,
        log_path_list=log_path_list,
        map_meta_data=args.meta.split(',')
    )

    if args.mode == 'ensemble':
        pp.ensemble_results()
    else:
        pp.concat_results()

    pp.calculate_metrics(
        metrics=['AUC', 'roc_auc', 'f1', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta'],
        metrics_sources=['tf', 'sklearn', 'sklearn',  'tf', 'tf', 'tf'],
        process_functions=[None, None, binarize, None, None, None]
    )
