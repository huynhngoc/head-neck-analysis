from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator

from deoxys.loaders import load_data
from deoxys.customize import custom_datareader
from deoxys.utils import file_finder, load_json_config
from deoxys.experiment import Experiment
from deoxys.model.callbacks import PredictionCheckpoint

import numpy as np
import h5py
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
from itertools import product
import pandas as pd
import os


@custom_datareader
class H5PatchReader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold',
                 patch_size=128, overlap=0.5, shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0.1, check_drop_channel=None):
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5_filename

        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.shuffle = shuffle

        self.patch_size = patch_size
        self.overlap = overlap

        self.preprocess_first = preprocess_first
        self.drop_fraction = drop_fraction
        self.check_drop_channel = check_drop_channel

        self.preprocessors = preprocessors
        self.augmentations = augmentations

        if preprocessors:
            if '__iter__' not in dir(preprocessors):
                self.preprocessors = [preprocessors]

        if augmentations:
            if '__iter__' not in dir(augmentations):
                self.augmentations = [augmentations]

        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        train_folds = list(train_folds) if train_folds else [0]
        test_folds = list(test_folds) if test_folds else [2]
        val_folds = list(val_folds) if val_folds else [1]

        if fold_prefix:
            self.train_folds = ['{}_{}'.format(
                fold_prefix, train_fold) for train_fold in train_folds]
            self.test_folds = ['{}_{}'.format(
                fold_prefix, test_fold) for test_fold in test_folds]
            self.val_folds = ['{}_{}'.format(
                fold_prefix, val_fold) for val_fold in val_folds]
        else:
            self.train_folds = train_folds
            self.test_folds = test_folds
            self.val_folds = val_folds

        self._original_test = None
        self._original_val = None

    @property
    def train_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=self.shuffle,
            augmentations=self.augmentations,
            preprocess_first=self.preprocess_first,
            drop_fraction=self.drop_fraction,
            check_drop_channel=self.check_drop_channel)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)

    @property
    def original_test(self):
        """
        Return a dictionary of all data in the test set
        """
        if self._original_test is None:
            self._original_test = {}
            for key in self.hf[self.test_folds[0]].keys():
                data = None
                for fold in self.test_folds:
                    new_data = self.hf[fold][key][:]

                    if data is None:
                        data = new_data
                    else:
                        data = np.concatenate((data, new_data))
                self._original_test[key] = data

        return self._original_test

    @property
    def original_val(self):
        """
        Return a dictionary of all data in the val set
        """
        if self._original_val is None:
            self._original_val = {}
            for key in self.hf[self.val_folds[0]].keys():
                data = None
                for fold in self.val_folds:
                    new_data = self.hf[fold][key][:]

                    if data is None:
                        data = new_data
                    else:
                        data = np.concatenate((data, new_data))
                self._original_val[key] = data

        return self._original_val


class H5PatchGenerator(DataGenerator):
    def __init__(self, h5_filename, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y',
                 folds=None,
                 patch_size=128, overlap=0.5,
                 shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0,
                 check_drop_channel=None):

        if not folds or not h5_filename:
            raise ValueError("h5file or folds is empty")

        # Checking for existence of folds and dataset
        with h5py.File(h5_filename, 'r') as h5file:
            group_names = h5file.keys()
            dataset_names = []
            str_folds = [str(fold) for fold in folds]
            for fold in str_folds:
                if fold not in group_names:
                    raise RuntimeError(
                        'HDF5 file: Fold name "{0}" is not in this h5 file'
                        .format(fold))
                if dataset_names:
                    if h5file[fold].keys() != dataset_names:
                        raise RuntimeError(
                            'HDF5 file: All folds should have the same structure')
                else:
                    dataset_names = h5file[fold].keys()
                    if x_name not in dataset_names or y_name not in dataset_names:
                        raise RuntimeError(
                            'HDF5 file: {0} or {1} is not in the file'
                            .format(x_name, y_name))

        # Checking for valid preprocessor
        if preprocessors:
            # if type(preprocessors) == list:
            for pp in preprocessors:
                if not callable(getattr(pp, 'transform', None)):
                    raise ValueError(
                        'Preprocessor should have a "transform" method')
            # else:
                # if not callable(getattr(preprocessors, 'transform', None)):
                #     raise ValueError(
                #         'Preprocessor should have a "transform" method')

        if augmentations:
            # if type(augmentations) == list:
            for pp in augmentations:
                if not callable(getattr(pp, 'transform', None)):
                    raise ValueError(
                        'Augmentation must be a preprocessor with'
                        ' a "transform" method')
            # else:
            #     if not callable(getattr(augmentations, 'transform', None)):
            #         raise ValueError(
            #             'Augmentation must be a preprocessor with'
            #             ' a "transform" method')

        self.h5_filename = h5_filename

        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.patch_size = patch_size
        self.overlap = overlap

        self.preprocessors = preprocessors
        self.augmentations = augmentations

        self.x_name = x_name
        self.y_name = y_name

        self.shuffle = shuffle
        self.preprocess_first = preprocess_first
        self.drop_fraction = drop_fraction
        self.check_drop_channel = check_drop_channel

        self.folds = str_folds

        self._total_batch = None

        # initialize "index" of current seg and fold
        self.seg_idx = 0
        self.fold_idx = 0

        # shuffle the folds
        if self.shuffle:
            np.random.shuffle(self.folds)

        # calculate number of segs in this fold
        with h5py.File(self.h5_filename, 'r') as h5file:
            seg_num = np.ceil(
                h5file[self.folds[0]][y_name].shape[0] / self.batch_cache)
            self.fold_shape = h5file[self.folds[0]][y_name].shape[1:-1]

        self.seg_list = np.arange(seg_num).astype(int)
        if self.shuffle:
            np.random.shuffle(self.seg_list)

        # fix patch_size if an int
        if '__iter__' not in dir(self.patch_size):
            self.patch_size = [patch_size] * len(self.fold_shape)

    def _apply_preprocess(self, x, y):
        seg_x, seg_y = x, y

        for preprocessor in self.preprocessors:
            seg_x, seg_y = preprocessor.transform(
                seg_x, seg_y)

        return seg_x, seg_y

    def _apply_augmentation(self, x, y):
        seg_x, seg_y = x, y

        for preprocessor in self.augmentations:
            seg_x, seg_y = preprocessor.transform(
                seg_x, seg_y)

        return seg_x, seg_y

    @property
    def total_batch(self):
        """Total number of batches to iterate all data.
        It will be used as the number of steps per epochs when training or
        validating data in a model.

        Returns
        -------
        int
            Total number of batches to iterate all data
        """
        print('counting total iter')
        if self._total_batch is None:
            total_batch = 0
            fold_names = self.folds

            if self.drop_fraction == 0:
                # just calculate based on the size of each fold
                for fold_name in fold_names:
                    with h5py.File(self.h5_filename, 'r') as hf:
                        shape = hf[fold_name][self.y_name].shape[:-1]
                    indices = get_patch_indice(
                        shape[1:], self.patch_size, self.overlap)
                    patches_per_img = len(indices)
                    patches_per_cache = patches_per_img * self.batch_cache

                    num_cache = shape[0] // self.batch_cache
                    remainder_img = shape[0] % self.batch_cache

                    total_batch += num_cache * patches_per_cache
                    total_batch += remainder_img * patches_per_img
            else:
                # have to apply preprocessor, if any before calculating
                # number of patches per image
                for fold_name in fold_names:
                    print(fold_name)
                    with h5py.File(self.h5_filename, 'r') as hf:
                        shape = hf[fold_name][self.y_name].shape[:-1]

                    indices = get_patch_indice(
                        shape[1:], self.patch_size, self.overlap)
                    for i in range(0, shape[0], self.batch_cache):
                        with h5py.File(self.h5_filename, 'r') as hf:
                            cache_x = hf[fold_name][
                                self.x_name][i: i + self.batch_cache]
                            cache_y = hf[fold_name][
                                self.y_name][i: i + self.batch_cache]
                        if self.preprocessors and self.preprocess_first:
                            cache_x, cache_y = self._apply_preprocess(
                                cache_x, cache_y)
                        # patches = get_patches(cache_x, target=None,
                        #             patch_indice=indices,
                        #             patch_size=self.patch_size,
                        #             stratified=False,
                        #             batch_size=self.batch_size,
                        #             drop_fraction=self.drop_fraction,
                        #             check_drop_channel=self.check_drop_channel)

                        patch_indice = np.array(
                            list((product(
                                np.arange(cache_x.shape[0]), indices))),
                            dtype=object)

                        check_drop_list = check_drop(
                            cache_x, patch_indice, self.patch_size,
                            self.drop_fraction, self.check_drop_channel)

                        total_batch += np.sum(check_drop_list)

        self._total_batch = int(total_batch)
        print('done counting iter_num')
        return self._total_batch

    def next_fold(self):
        self.fold_idx += 1

        if self.fold_idx == len(self.folds):
            self.fold_idx = 0

            if self.shuffle:
                np.random.shuffle(self.folds)

    def next_seg(self):
        if self.seg_idx == len(self.seg_list):
            # move to next fold
            self.next_fold()

            # reset seg index
            self.seg_idx = 0
            # recalculate seg_num
            cur_fold = self.folds[self.fold_idx]
            with h5py.File(self.h5_filename, 'r') as hf:
                seg_num = np.ceil(
                    hf[cur_fold][self.y_name].shape[0] / self.batch_cache)
                self.fold_shape = hf[self.folds[0]][self.y_name].shape[1:-1]

            self.seg_list = np.arange(seg_num).astype(int)

            if self.shuffle:
                np.random.shuffle(self.seg_list)

        cur_fold = self.folds[self.fold_idx]
        cur_seg_idx = self.seg_list[self.seg_idx]

        start, end = cur_seg_idx * \
            self.batch_cache, (cur_seg_idx + 1) * self.batch_cache

        # print(cur_fold, cur_seg_idx, start, end)

        with h5py.File(self.h5_filename, 'r') as hf:
            seg_x_raw = hf[cur_fold][self.x_name][start: end]
            seg_y_raw = hf[cur_fold][self.y_name][start: end]

        indices = get_patch_indice(
            self.fold_shape, self.patch_size, self.overlap)

        # if preprocess first, apply preprocess here
        if self.preprocessors and self.preprocess_first:
            seg_x_raw, seg_y_raw = self._apply_preprocess(seg_x_raw, seg_y_raw)
        # get patches
        seg_x, seg_y = get_patches(
            seg_x_raw, seg_y_raw,
            patch_indice=indices, patch_size=self.patch_size,
            stratified=self.shuffle, batch_size=self.batch_size,
            drop_fraction=self.drop_fraction,
            check_drop_channel=self.check_drop_channel)

        # if preprocess after patch, apply preprocess here
        if self.preprocessors and not self.preprocess_first:
            seg_x, seg_y = self._apply_preprocess(seg_x, seg_y)

        # finally apply augmentation, if any
        if self.augmentations:
            seg_x, seg_y = self._apply_augmentation(seg_x, seg_y)

        # # if self.shuffle:
        # #     np.random.shuffle(return_indice)

        # # Apply preprocessor
        # if self.preprocessors:
        #     # if type(self.preprocessors) == list:
        #     for preprocessor in self.preprocessors:
        #         seg_x, seg_y = preprocessor.transform(
        #             seg_x, seg_y)
        #     # else:
        #     #     seg_x, seg_y = self.preprocessors.transform(
        #     #         seg_x, seg_y)
        # # Apply augmentation:
        # if self.augmentations:
        #     # if type(self.augmentations) == list:
        #     for preprocessor in self.augmentations:
        #         seg_x, seg_y = preprocessor.transform(
        #             seg_x, seg_y)
        #     # else:
        #     #     seg_x, seg_y = self.augmentations.transform(
        #     #         seg_x, seg_y)

        # increase seg index
        self.seg_idx += 1
        print(self.seg_idx, seg_x.shape, seg_y.shape)
        return seg_x, seg_y

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
        """
        while True:
            seg_x, seg_y = self.next_seg()

            seg_len = len(seg_y)

            for i in range(0, seg_len, self.batch_size):
                batch_x = seg_x[i:(i + self.batch_size)]
                batch_y = seg_y[i:(i + self.batch_size)]

                # print(batch_x.shape)

                yield batch_x, batch_y


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

    TEST_SINGLE_MAP_NAME = '/single_result.csv'
    TEST_MAP_NAME = 'single_result.csv'

    def __init__(self, log_base_path='logs',
                 temp_base_path='', map_meta_data=None, main_meta_data='', run_test=False):
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

        self.run_test = run_test

    def map_2d_meta_data(self):
        print('mapping 2d meta data')
        if not self.run_test:
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
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME
            H5MetaDataMapping(
                ref_file=self.dataset_filename,
                save_file=map_filename,
                folds=self.data_reader.test_folds,
                fold_prefix='',
                dataset_names=self.map_meta_data).post_process()

        return self

    def calculate_fscore_single(self):
        print('calculating dice score per items in val set')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            map_filename = map_folder + self.SINGLE_MAP_NAME
            for epoch in self.epochs:
                H5CalculateFScore(
                    predicted_path.format(epoch=epoch),
                    map_filename.format(epoch=epoch)
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME

            H5CalculateFScore(
                predicted_path,
                map_filename
            ).post_process()

        return self

    def merge_2d_slice(self):
        print('merge 2d slice to 3d images')
        if not self.run_test:
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
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            H5Merge2dSlice(
                predicted_path,
                map_filename,
                self.main_meta_data,
                merge_path,
                main_result_file_name
            ).post_process()

        return self

    def calculate_fscore(self):
        print('calculating dice score per 3d image')
        if not self.run_test:
            merge_path = self.log_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5CalculateFScore(
                    merge_path.format(epoch=epoch),
                    main_log_filename.format(epoch=epoch),
                    map_file=main_log_filename.format(epoch=epoch),
                    map_column=self.main_meta_data
                ).post_process()
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            H5CalculateFScore(
                merge_path,
                main_result_file_name.format,
                map_file=main_result_file_name,
                map_column=self.main_meta_data
            ).post_process()

        return self

    def get_best_model(self, monitor='', keep_best_only=True):
        print('finding best model')

        epochs = self.epochs

        res_df = pd.DataFrame(epochs, columns=['epochs'])

        results = []
        results_path = self.log_base_path + self.MAP_PATH + self.MAP_NAME

        for epoch in epochs:
            df = pd.read_csv(results_path.format(epoch=epoch))
            if not monitor:
                monitor = df.columns[-1]

            results.append(df[monitor].mean())

        best_epoch = epochs[df[monitor].argmax()]
        res_df[monitor] = results

        res_df.to_csv(self.log_base_path + '/log_new.csv', index=False)

        if keep_best_only:
            for epoch in epochs:
                if epoch != best_epoch:
                    os.remove(self.log_base_path + self.PREDICTION_PATH +
                              self.PREDICTION_NAME.format(epoch=epoch))

        return self.log_base_path + self.MODEL_PATH + \
            self.MODEL_NAME.format(epoch=best_epoch)


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
