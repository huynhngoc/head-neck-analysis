"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import Experiment
from deoxys.utils import read_file
import argparse
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import customize_obj

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=25, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=25, type=int)

    args = parser.parse_args()

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    config = read_file(args.config_file)

    exp = Experiment(
        log_base_path=args.log_folder
    ).from_full_config(
        config
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
    )

    # Find early stopping epoch, if any
    train_params = exp.model._get_train_params
    if 'callbacks' in train_params:
        stopped_epoch = 0
        for callback in train_params['callback']:
            if isinstance(callback, EarlyStopping):
                stopped_epoch = callback.stopped_epoch
                break

        if stopped_epoch > 0:
            exp.run_experiment(
                train_history_log=True,
                model_checkpoint_period=1,
                prediction_checkpoint_period=1,
                epochs=stopped_epoch + 2,
                initial_epoch=stopped_epoch + 1
            )

    # 42 images for 2d, 15 images for 3d
    img_num = 42

    if '3d' in args.log_folder:
        img_num = 15

    # Plot performance
    exp.plot_performance().plot_prediction(
        masked_images=[i for i in range(img_num)])
