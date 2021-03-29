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
import shutil
import gc
import time

# from threading import Thread
import threading
from multiprocessing import Queue, Process


@custom_datareader
class H5PatchReader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold',
                 patch_size=128, overlap=0.5, shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0.1, check_drop_channel=None,
                 bounding_box=False):
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
        self.bounding_box = bounding_box

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
            check_drop_channel=self.check_drop_channel,
            bounding_box=self.bounding_box)

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
                 check_drop_channel=None,
                 bounding_box=False):

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
        self.bounding_box = bounding_box

        self.folds = str_folds

        self._total_batch = None
        self._description = None

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

        if self.augmentations:
            self.queue = Queue(10)
            self.running_process = Process(target=self._next_seg)
            self.running_process.daemon = True
            self.running_process.start()
            # sleep for 2 mins to buffer queue
            time.sleep(120)

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
    def description(self):
        if self.shuffle:
            raise Warning('The data is shuffled, the description results '
                          'may not be accurate')
        if self._description is None:
            fold_names = self.folds
            description = []
            # find the shape of the inputs in the first fold
            with h5py.File(self.h5_filename, 'r') as hf:
                shape = hf[fold_names[0]][self.x_name].shape
            obj = {'shape': shape[1:], 'total': shape[0]}

            for fold_name in fold_names[1:]:  # iterate through each fold
                with h5py.File(self.h5_filename, 'r') as hf:
                    shape = hf[fold_name][self.x_name].shape

                # if the shape are the same, increase the total number
                if np.all(obj['shape'] == shape[1:]):
                    obj['total'] += shape[0]
                # else create a new item
                else:
                    description.append(obj.copy())
                    obj = {'shape': shape[1:], 'total': shape[0]}

            # append the last item
            description.append(obj.copy())

            final_shape = self.patch_size
            if len(self.patch_size) < len(obj['shape']):
                final_shape = final_shape + \
                    list(obj['shape'][len(final_shape):])

            final_shape = tuple(final_shape)

            final_obj = {'shape': final_shape, 'total': 0}
            for obj in description:
                indice = get_patch_indice(
                    obj['shape'][:len(self.patch_size)],
                    self.patch_size, self.overlap)
                final_obj['total'] += obj['total'] * len(indice)

            self._description = [final_obj]
        return self._description

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

                    batch_per_cache = np.ceil(
                        patches_per_cache / self.batch_size)

                    total_batch += num_cache * batch_per_cache

                    total_batch += np.ceil(
                        remainder_img * patches_per_img / self.batch_size)

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

                        patch_indice = np.array(
                            list((product(
                                np.arange(cache_x.shape[0]), indices))),
                            dtype=object)

                        if self.bounding_box:
                            check_drop_list = check_drop(
                                cache_y, patch_indice, self.patch_size,
                                self.drop_fraction, self.check_drop_channel)
                        else:
                            check_drop_list = check_drop(
                                cache_x, patch_indice, self.patch_size,
                                self.drop_fraction, self.check_drop_channel)

                        total_batch += np.ceil(
                            np.sum(check_drop_list) / self.batch_size)

        self._total_batch = int(total_batch)
        print('done counting iter_num', self._total_batch)
        return self._total_batch

    def next_fold(self):
        self.fold_idx += 1

        if self.fold_idx == len(self.folds):
            self.fold_idx = 0

            if self.shuffle:
                np.random.shuffle(self.folds)

    def next_seg(self):
        gc.collect()
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
            check_drop_channel=self.check_drop_channel,
            bounding_box=self.bounding_box)

        # if preprocess after patch, apply preprocess here
        if self.preprocessors and not self.preprocess_first:
            seg_x, seg_y = self._apply_preprocess(seg_x, seg_y)

        # finally apply augmentation, if any
        # if self.augmentations:
        #     seg_x, seg_y = self._apply_augmentation(seg_x, seg_y)

        # increase seg index
        self.seg_idx += 1
        return seg_x, seg_y

    def _next_seg(self):
        while True:
            seg_x, seg_y = self.next_seg()
            seg_len = len(seg_y)
            i = 0
            if not self.queue.full() and i < seg_len:
                print('Putting item into queue', self.queue.qsize())
                batch_x = seg_x[i:(i + self.batch_size)]
                batch_y = seg_y[i:(i + self.batch_size)]
                self._apply_augmentation(batch_x, batch_y)
                self.queue.put((batch_x, batch_y))
                i += self.batch_size

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
        """
        if self.augmentations:
            while True:
                if not self.queue.empty():
                    batch_x, batch_y = self.queue.get()  # next_seg()
                    yield batch_x, batch_y

                    # seg_len = len(seg_y)

                    # for i in range(0, seg_len, self.batch_size):
                    #     batch_x = seg_x[i:(i + self.batch_size)]
                    #     batch_y = seg_y[i:(i + self.batch_size)]
                    #     yield batch_x, batch_y
        else:
            while True:
                seg_x, seg_y = self.next_seg()
                seg_len = len(seg_y)

                for i in range(0, seg_len, self.batch_size):
                    batch_x = seg_x[i:(i + self.batch_size)]
                    batch_y = seg_y[i:(i + self.batch_size)]
                    yield batch_x, batch_y

    def __del__(self):
        if self.augmentations:
            self.queue.close()
            self.running_process.terminate()
