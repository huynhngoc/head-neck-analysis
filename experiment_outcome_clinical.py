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
from deoxys.utils import read_csv
import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()


def metric_avg_score(res_df, postprocessor):
    auc = res_df['AUC']
    mcc = res_df['mcc'] / 2 + 0.5
    f1 = res_df['f1']
    f0 = res_df['f1_0']

    # get f1 score in train data
    epochs = res_df['epochs']
    train_df = read_csv(
        postprocessor.log_base_path + '/logs.csv')
    train_df['real_epoch'] = train_df['epoch'] + 1
    train_f1 = train_df[train_df.real_epoch.isin(epochs)]['BinaryFbeta'].values
    train_f1 = 2 * np.sqrt(train_f1) / 3

    res_df['avg_score'] = (auc + mcc + f1 + 0.8*f0 + 0.8*train_f1) / 4.6

    return res_df


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
        "--monitor", default='avg_score', type=str)
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

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    class_weight = None
    if 'LRC' in args.log_folder:
        class_weight = {0: 0.3, 1: 0.7}

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_full_config(
        args.config_file
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=20,
        prediction_checkpoint_period=20,
        epochs=40,
        save_val_inputs=False,
        class_weight=class_weight,
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
        initial_epoch=40,
        save_val_inputs=False,
    ).apply_post_processors(
        map_meta_data=meta,
        metrics=['AUC', 'roc_auc', 'f1', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta', 'mcc', 'f1'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'tf', 'sklearn', 'sklearn'],
        process_functions=[None, None, binarize, None, None, None, binarize,
                           flip],
        metrics_kwargs=[{}, {}, {}, {}, {}, {}, {}, {'metric_name': 'f1_0'}]
    ).plot_performance().load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    ).run_test(
    ).apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['AUC', 'roc_auc', 'f1', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta', 'mcc', 'f1'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'tf', 'sklearn', 'sklearn'],
        process_functions=[None, None, binarize, None, None, None, binarize,
                           flip],
        metrics_kwargs=[{}, {}, {}, {}, {}, {}, {}, {'metric_name': 'f1_0'}]
    )
