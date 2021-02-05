"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

import customize_obj
import h5py
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import Experiment
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.utils import read_file
import argparse
import os
import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx


class AnalysisExperiment(Experiment):
    def __init__(self,
                 log_base_path='logs',
                 temp_base_path='',
                 best_model_monitors='val_loss',
                 best_model_modes='auto'):

        self.temp_base_path = temp_base_path

        super().__init__(log_base_path, best_model_monitors, best_model_modes)

    def _create_prediction_checkpoint(self, base_path, period, use_original):
        temp_base_path = self.temp_base_path
        if temp_base_path:
            if not os.path.exists(temp_base_path):
                os.makedirs(temp_base_path)

            if not os.path.exists(temp_base_path + self.PREDICTION_PATH):
                os.makedirs(temp_base_path + self.PREDICTION_PATH)

        pred_base_path = temp_base_path or base_path

        return PredictionCheckpoint(
            filepath=pred_base_path + self.PREDICTION_PATH + self.PREDICTION_NAME,
            period=period, use_original=use_original)

    def plot_prediction(self, masked_images,
                        contour=True,
                        base_image_name='x',
                        truth_image_name='y',
                        predicted_image_title_name='Image {index:05d}',
                        img_name='{index:05d}.png'):
        log_base_path = self.temp_base_path or self.log_base_path

        if os.path.exists(log_base_path + self.PREDICTION_PATH):
            print('\nCreating prediction images...')
            # mask images
            prediced_image_path = log_base_path + self.PREDICTED_IMAGE_PATH
            if not os.path.exists(prediced_image_path):
                os.makedirs(prediced_image_path)

            for filename in os.listdir(
                    log_base_path + self.PREDICTION_PATH):
                if filename.endswith(".h5") or filename.endswith(".hdf5"):
                    # Create a folder for storing result in that period
                    images_path = prediced_image_path + '/' + filename
                    if not os.path.exists(images_path):
                        os.makedirs(images_path)

                    self._plot_predicted_images(
                        data_path=log_base_path + self.PREDICTION_PATH
                        + '/' + filename,
                        out_path=images_path,
                        images=masked_images,
                        base_image_name=base_image_name,
                        truth_image_name=truth_image_name,
                        title=predicted_image_title_name,
                        contour=contour,
                        name=img_name)

        return self

    def run_test(self, use_best_model=False,
                 masked_images=None,
                 use_original_image=False,
                 contour=True,
                 base_image_name='x',
                 truth_image_name='y',
                 image_name='{index:05d}.png',
                 image_title_name='Image {index:05d}'):

        log_base_path = self.temp_base_path or self.log_base_path

        test_path = log_base_path + self.TEST_OUTPUT_PATH

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if use_best_model:
            raise NotImplementedError
        else:
            score = self.model.evaluate_test(verbose=1)
            print(score)

            predicted = self.model.predict_test(verbose=1)

            # Create the h5 file
            filepath = test_path + self.PREDICT_TEST_NAME
            hf = h5py.File(filepath, 'w')
            hf.create_dataset('predicted', data=predicted)
            hf.close()

            if use_original_image:
                original_data = self.model.data_reader.original_test

                for key, val in original_data.items():
                    hf = h5py.File(filepath, 'a')
                    hf.create_dataset(key, data=val)
                    hf.close()
            else:
                # Create data from test_generator
                x = None
                y = None

                test_gen = self.model.data_reader.test_generator
                data_gen = test_gen.generate()

                for _ in range(test_gen.total_batch):
                    next_x, next_y = next(data_gen)
                    if x is None:
                        x = next_x
                        y = next_y
                    else:
                        x = np.concatenate((x, next_x))
                        y = np.concatenate((y, next_y))

                hf = h5py.File(filepath, 'a')
                hf.create_dataset('x', data=x)
                hf.create_dataset('y', data=y)
                hf.close()

            if masked_images:
                self._plot_predicted_images(
                    data_path=filepath,
                    out_path=test_path,
                    images=masked_images,
                    base_image_name=base_image_name,
                    truth_image_name=truth_image_name,
                    title=image_title_name,
                    contour=contour,
                    name=image_name)

        return self


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=25, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=25, type=int)

    args = parser.parse_args()

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    config = read_file(args.config_file)

    exp = AnalysisExperiment(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_full_config(
        config
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
    )

    # Find early stopping epoch, if any
    train_params = exp.model._get_train_params(['callbacks'])
    # exp.model._train_params # alternative
    print(train_params)
    if 'callbacks' in train_params:
        stopped_epoch = 0
        for callback in train_params['callbacks']:
            print(callback)
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
