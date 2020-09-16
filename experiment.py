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

    config = read_file(args.config_file)
    (
        Experiment(log_base_path=args.log_folder)
        .from_full_config(config)
        .run_experiment(
            train_history_log=True,
            model_checkpoint_period=args.model_checkpoint_period,
            prediction_checkpoint_period=args.prediction_checkpoint_period,
            epochs=args.epochs,
        ).run_experiment(
            train_history_log=True,
            model_checkpoint_period=1,
            prediction_checkpoint_period=1,
            epochs=args.epochs + 1,
            initial_epoch=args.epochs
        )
        .plot_performance()
        .plot_prediction(masked_images=[i for i in range(42)])
    )
