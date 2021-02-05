# from deoxys.experiment import Experiment
# from deoxys.utils import read_file
import argparse
import numpy as np
import h5py
import pandas as pd
import os

from deoxys.utils import load_json_config
from deoxys.loaders import load_data
# from pathlib import Path
# from comet_ml import Experiment as CometEx
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# import customize_obj


class H5Metric:
    def __init__(self, ref_file, save_file, metric_name='score',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4,
                 map_file=None, map_column=None):
        self.metric_name = metric_name
        self.ref_file = ref_file

        self.predicted = predicted_dataset
        self.target = target_dataset
        self.batch_size = batch_size

        self.res_file = save_file
        self.map_file = map_file
        self.map_column = map_column

    def get_img_batch(self):
        self.scores = []

        if self.map_file is None:
            with h5py.File(self.ref_file, 'r') as f:
                size = f[self.target].shape[0]

            for i in range(0, size, self.batch_size):
                with h5py.File(self.ref_file, 'r') as f:
                    predicted = f[self.predicted][i:i+self.batch_size]
                    targets = f[self.target][i:i+self.batch_size]
                yield targets, predicted
        else:  # handle 3d with different sizes
            map_df = pd.read_csv(self.map_file)
            map_data = map_df[self.map_column].values

            for idx in map_data:
                with h5py.File(self.ref_file, 'r') as f:
                    predicted = f[self.predicted][str(idx)][:]
                    targets = f[self.target][str(idx)][:]
                yield np.expand_dims(targets, axis=0), np.expand_dims(predicted, axis=0)

    def update_score(self, scores):
        self.scores.extend(scores)

    def save_score(self):
        if os.path.isfile(self.res_file):
            df = pd.read_csv(self.res_file)
            df[f'{self.metric_name}'] = self.scores
        else:
            df = pd.DataFrame(self.scores, columns=[f'{self.metric_name}'])

        df.to_csv(self.res_file, index=False)

    def post_process(self, **kwargs):
        for targets, prediction in self.get_img_batch():
            scores = self.calculate_metrics(
                targets, prediction, **kwargs)
            self.update_score(scores)

        self.save_score()

    def calculate_metrics(targets, predictions, **kwargs):
        raise NotImplementedError


class H5CalculateFScore(H5Metric):
    def __init__(self, ref_file, save_file, metric_name='f1_score',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None):
        super().__init__(ref_file, save_file, metric_name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        assert len(y_true) == len(y_pred), "Shape not match"
        eps = 1e-8
        size = len(y_true.shape)
        reduce_ax = tuple(range(1, size))

        y_pred = (y_pred > self.threshold).astype(y_pred.dtype)

        true_positive = np.sum(y_pred * y_true, axis=reduce_ax)
        target_positive = np.sum(y_true, axis=reduce_ax)
        predicted_positive = np.sum(y_pred, axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return fb_numerator / fb_denominator


class H5MetaDataMapping:
    def __init__(self, ref_file, save_file, folds, fold_prefix='fold',
                 dataset_names=None):
        self.ref_file = ref_file
        self.save_file = save_file
        if fold_prefix:
            self.folds = ['{}_{}'.format(
                fold_prefix, fold) for fold in folds]
        else:
            self.folds = folds

        self.dataset_names = dataset_names

    def post_process(self, *args, **kwargs):
        data = {dataset_name: [] for dataset_name in self.dataset_names}
        for fold in self.folds:
            with h5py.File(self.ref_file, 'r') as f:
                for dataset_name in self.dataset_names:
                    data[dataset_name].extend(f[fold][dataset_name][:])

        df = pd.DataFrame(data)
        df.to_csv(self.save_file, index=False)


class H5Merge2dSlice:
    def __init__(self, ref_file, map_file, map_column, merge_file, save_file,
                 predicted_dataset='predicted', target_dataset='y',
                 input_dataset='x'):
        self.ref_file = ref_file
        self.map_file = map_file
        self.map_column = map_column
        self.merge_file = merge_file
        self.save_file = save_file

        self.predicted = predicted_dataset
        self.target = target_dataset
        self.inputs = input_dataset

    def post_process(self):
        map_df = pd.read_csv(self.map_file)
        map_data = map_df[self.map_column].values

        unique_val = []

        first, last = map_data[0], map_data[-1]

        tmp = np.concatenate([[first], map_data, [last]])
        indice = np.where(tmp[1:] != tmp[:-1])[0]
        indice = np.concatenate([[0], indice, [len(map_data)]])

        with h5py.File(self.merge_file, 'w') as mf:
            mf.create_group(self.inputs)
            mf.create_group(self.target)
            mf.create_group(self.predicted)

        for i in range(len(indice) - 1):
            start = indice[i]
            end = indice[i+1]

            unique_val.append(map_data[start])

            assert map_data[start] == map_data[end-1], "id not match"

            curr_name = str(map_data[start])
            with h5py.File(self.ref_file, 'r') as f:
                img = f[self.inputs][start:end]
            with h5py.File(self.merge_file, 'a') as mf:
                mf[self.inputs].create_dataset(
                    curr_name, data=img, compression="gzip")

            with h5py.File(self.ref_file, 'r') as f:
                img = f[self.target][start:end]
            with h5py.File(self.merge_file, 'a') as mf:
                mf[self.target].create_dataset(
                    curr_name, data=img, compression="gzip")

            with h5py.File(self.ref_file, 'r') as f:
                img = f[self.predicted][start:end]
            with h5py.File(self.merge_file, 'a') as mf:
                mf[self.predicted].create_dataset(
                    curr_name, data=img, compression="gzip")

        df = pd.DataFrame(data=unique_val, columns=[self.map_column])
        df.to_csv(self.save_file, index=False)


class PostProcessor:
    MODEL_PATH = '/model'
    MODEL_NAME = '/model.{epoch:03d}.h5'
    BEST_MODEL_PATH = '/best'
    PREDICTION_PATH = '/prediction'
    PREDICTION_NAME = '/prediction.{epoch:03d}.h5'
    LOG_FILE = '/logs.csv'
    PERFORMANCE_PATH = '/performance'
    PREDICTED_IMAGE_PATH = '/images'
    TEST_OUTPUT_PATH = '/test'
    PREDICT_TEST_NAME = '/prediction_test.h5'
    SINGLE_MAP_PATH = '/single_map'
    SINGLE_MAP_NAME = '/logs.{epoch:03d}.csv'

    MAP_PATH = '/logs'
    MAP_NAME = '/logs.{epoch:03d}.csv'

    def __init__(self, log_base_path='logs',
                 temp_base_path='', map_meta_data=None, main_meta_data=''):
        self.temp_base_path = temp_base_path
        self.log_base_path = log_base_path

        model_path = log_base_path + self.MODEL_PATH

        sample_model_filename = model_path + '/' + os.listdir(model_path)[0]

        with h5py.File(sample_model_filename, 'r') as f:
            config = f.attrs['deoxys_config']
            config = load_json_config(config)

        self.dataset_filename = config['dataset_params']['config']['filename']
        self.data_reader = load_data(config['dataset_params'])

        self.temp_prediction_path = temp_base_path + self.PREDICTION_PATH
        predicted_files = os.listdir(self.temp_prediction_path)

        self.epochs = [int(filename[-6:-3]) for filename in predicted_files]

        if map_meta_data:
            self.map_meta_data = map_meta_data.split(',')
        else:
            self.map_meta_data = ['patient_idx', 'slice_idx']

        if main_meta_data:
            self.main_meta_data = main_meta_data
        else:
            self.main_meta_data = self.map_meta_data[0]

    def map_2d_meta_data(self):
        print('mapping 2d meta data')
        map_folder = self.log_base_path + self.SINGLE_MAP_PATH

        if not os.path.exists(map_folder):
            os.makedirs(map_folder)
        map_filename = map_folder + self.SINGLE_MAP_NAME

        for epoch in self.epochs:
            H5MetaDataMapping(
                ref_file=self.dataset_filename,
                save_file=map_filename.format(epoch=epoch),
                folds=self.data_reader.val_folds,
                fold_prefix='',
                dataset_names=self.map_meta_data).post_process()

        return self

    def calculate_fscore_single(self):
        print('calculating dice score per items in val set')
        predicted_path = self.temp_base_path + \
            self.PREDICTION_PATH + self.PREDICTION_NAME
        map_folder = self.log_base_path + self.SINGLE_MAP_PATH
        map_filename = map_folder + self.SINGLE_MAP_NAME
        for epoch in self.epochs:
            H5CalculateFScore(
                predicted_path.format(epoch=epoch),
                map_filename.format(epoch=epoch)
            ).post_process()

        return self

    def merge_2d_slice(self):
        print('merge 2d slice to 3d images')
        predicted_path = self.temp_base_path + \
            self.PREDICTION_PATH + self.PREDICTION_NAME
        map_folder = self.log_base_path + self.SINGLE_MAP_PATH
        map_filename = map_folder + self.SINGLE_MAP_NAME

        merge_path = self.log_base_path + \
            self.PREDICTION_PATH + self.PREDICTION_NAME

        main_log_folder = self.log_base_path + self.MAP_PATH

        if not os.path.exists(main_log_folder):
            os.makedirs(main_log_folder)
        main_log_filename = main_log_folder + self.MAP_NAME

        for epoch in self.epochs:
            H5Merge2dSlice(
                predicted_path.format(epoch=epoch),
                map_filename.format(epoch=epoch),
                self.main_meta_data,
                merge_path.format(epoch=epoch),
                main_log_filename.format(epoch=epoch)
            ).post_process()
        return self

    def calculate_fscore(self):
        print('calculating dice score per 3d image')
        merge_path = self.log_base_path + \
            self.PREDICTION_PATH + self.PREDICTION_NAME

        main_log_folder = self.log_base_path + self.MAP_PATH
        main_log_filename = main_log_folder + self.MAP_NAME

        for epoch in self.epochs:
            H5CalculateFScore(
                merge_path.format(epoch=epoch),
                main_log_folder.format(epoch=epoch),
                map_file=main_log_filename.format(epoch=epoch),
                map_column=self.main_meta_data
            ).post_process()

        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder",
                        default='', type=str)
    parser.add_argument("--meta", default='patient_idx,slice_idx', type=str)

    args = parser.parse_args()

    PostProcessor(
        args.log_folder,
        temp_base_path=args.temp_folder,
        map_meta_data=args.meta
    ).map_2d_meta_data().calculate_fscore_single().merge_2d_slice(
    ).calculate_fscore()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file")  # prediction
    # parser.add_argument("log_folder")
    # parser.add_argument("--single_map_file",
    #                     default='single_map.csv', type=str)
    # parser.add_argument("--map_file", default='map.csv', type=str)
    # parser.add_argument("--map_column", default='patient_idx', type=str)
    # parser.add_argument("--columns", default='patient_idx,slice_idx', type=str)
    # parser.add_argument("--merge_file", default='merge_file.h5', type=str)
    # parser.add_argument("--folds", default='val', type=str)
    # parser.add_argument("--prefix", default='', type=str)

    # args = parser.parse_args()

    # print('mapping meta data')
    # print(args.folds)

    # # H5MetaDataMapping('/home/work/ngochuyn/hn_delin/full_dataset_singleclass.h5',
    # H5MetaDataMapping('../../full_dataset_singleclass.h5',
    #                   f'{args.log_folder}/{args.single_map_file}',
    #                   folds=args.folds.split(','), fold_prefix=args.prefix,
    #                   dataset_names=args.columns.split(',')).post_process()

    # print('calculate fscore meta data')

    # H5CalculateFScore(args.config_file,
    #                   f'{args.log_folder}/{args.single_map_file}').post_process()

    # print('merge data')

    # H5Merge2dSlice(args.config_file, f'{args.log_folder}/{args.single_map_file}',
    #                args.map_column, f'{args.log_folder}/{args.merge_file}',
    #                f'{args.log_folder}/{args.map_file}').post_process()

    # print('calculate dice')

    # H5CalculateFScore(f'{args.log_folder}/{args.merge_file}',
    #                   f'{args.log_folder}/{args.map_file}',
    #                   map_file=f'{args.log_folder}/{args.map_file}',
    #                   map_column=args.map_column).post_process()
