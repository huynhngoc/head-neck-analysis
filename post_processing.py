# from deoxys.experiment import Experiment
# from deoxys.utils import read_file
import argparse
import numpy as np
import h5py
import pandas as pd
import os
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

        with h5py.File(self.merge_file, 'a') as mf:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # prediction
    parser.add_argument("log_folder")
    parser.add_argument("--single_map_file",
                        default='single_map.csv', type=str)
    parser.add_argument("--map_file", default='map.csv', type=str)
    parser.add_argument("--map_column", default='patient_idx', type=str)
    parser.add_argument("--columns", default='patient_idx,slice_idx', type=str)
    parser.add_argument("--merge_file", default='merge_file.h5', type=str)
    parser.add_argument("--folds", default='val', type=str)
    parser.add_argument("--prefix", default='', type=str)

    args = parser.parse_args()

    print('mapping meta data')

    H5MetaDataMapping('/home/work/ngochuyn/hn_delin/full_dataset_singleclass.h5',
                      # H5MetaDataMapping('../../full_dataset_singleclass.h5',
                      f'{args.log_folder}/{args.single_map_file}',
                      folds=args.folds.split(','), fold_prefix=args.prefix,
                      dataset_names=args.columns.split(',')).post_process()

    print('calculate fscore meta data')

    H5CalculateFScore(args.config_file,
                      f'{args.log_folder}/{args.single_map_file}').post_process()

    print('merge data')

    H5Merge2dSlice(args.config_file, f'{args.log_folder}/{args.single_map_file}',
                   args.map_column, f'{args.log_folder}/{args.merge_file}',
                   f'{args.log_folder}/{args.map_file}').post_process()

    print('calculate dice')

    H5CalculateFScore(f'{args.log_folder}/{args.merge_file}',
                      f'{args.log_folder}/{args.map_file}',
                      map_file=f'{args.log_folder}/{args.map_file}',
                      map_column=args.map_column).post_process()
