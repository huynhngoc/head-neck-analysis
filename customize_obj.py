import gc
from itertools import product
import shutil
from deoxys.data.preprocessor import BasePreprocessor
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
import h5py
from tensorflow import image
from tensorflow.keras.layers import Input, concatenate, Lambda, \
    Add, Activation, Multiply
from tensorflow.keras.models import Model as KerasModel
import numpy as np
import tensorflow as tf
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.experiment import Experiment
from deoxys.experiment.postprocessor import DefaultPostProcessor
from deoxys.utils import file_finder, load_json_config
from deoxys.customize import custom_architecture, custom_datareader, custom_layer
from deoxys.loaders import load_data
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator
from deoxys.model.layers import layer_from_config
# from tensorflow.python.ops.gen_math_ops import square
import tensorflow_addons as tfa
from deoxys.model.losses import Loss, loss_from_config
from deoxys.customize import custom_loss, custom_preprocessor
from deoxys.data import ImageAugmentation2D
from elasticdeform import deform_random_grid
import new_layer
import os

multi_input_layers = ['Add', 'AddResize', 'Concatenate', 'Multiply']
resize_input_layers = ['Concatenate', 'AddResize']


@custom_preprocessor
class CroppedMask3D:
    def __init__(self, channel=-1, size=128):
        self.channel = channel
        self.size = size
        self.left = size // 2
        self.right = size - self.left

    def _get_bounding(self, mask, axis, max_size):
        args = np.argwhere(mask.sum(axis=axis) > 0).flatten()
        middle = (args.max() - args.min()) // 2
        left, right = middle - self.left, middle + self.right
        if left < 0:
            left = 0
            right = self.size
        if right > max_size:
            right = max_size
            left = right - self.size

            return left, right

    def transform(self, images, targets):
        masks = images[..., self.channel]
        shape = masks.shape[1:]
        new_images = []
        for i, mask in enumerate(masks):
            left_0, right_0 = self._get_bounding(
                mask, axis=(1, 2), max_size=shape[0])
            left_1, right_1 = self._get_bounding(
                mask, axis=(0, 2), max_size=shape[1])
            left_2, right_2 = self._get_bounding(
                mask, axis=(0, 1), max_size=shape[2])

            new_images.append(
                images[i][left_0: right_0, left_1: right_1, left_2: right_2]
            )

        return np.concatenate(new_images),  targets


@custom_layer
class InstanceNormalization(tfa.layers.InstanceNormalization):
    pass


@custom_layer
class AddResize(Add):
    pass


@custom_loss
class BinaryMacroFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_macro_fbeta",
                 beta=1, square=False):
        super().__init__(reduction, name)

        self.beta = beta
        self.square = square

    def call(self, target, prediction):
        eps = 1e-8
        target = tf.cast(target, prediction.dtype)

        true_positive = tf.math.reduce_sum(prediction * target)
        if self.square:
            target_positive = tf.math.reduce_sum(tf.math.square(target))
            predicted_positive = tf.math.reduce_sum(
                tf.math.square(prediction))
        else:
            target_positive = tf.math.reduce_sum(target)
            predicted_positive = tf.math.reduce_sum(prediction)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return 1 - fb_numerator / fb_denominator


@custom_loss
class FusedLoss(Loss):
    """Used to sum two or more loss functions.
    """

    def __init__(
            self, loss_configs, loss_weights=None,
            reduction="auto", name="fused_loss"):
        super().__init__(reduction, name)
        self.losses = [loss_from_config(loss_config)
                       for loss_config in loss_configs]

        if loss_weights is None:
            loss_weights = [1] * len(self.losses)
        self.loss_weights = loss_weights

    def call(self, target, prediction):
        loss = None
        for loss_class, loss_weight in zip(self.losses, self.loss_weights):
            if loss is None:
                loss = loss_weight * loss_class(target, prediction)
            else:
                loss += loss_weight * loss_class(target, prediction)

        return loss


class EnsemblePostProcessor(DefaultPostProcessor):
    def __init__(self, log_base_path='logs',
                 log_path_list=None,
                 map_meta_data=None, **kwargs):

        self.log_base_path = log_base_path
        self.log_path_list = []
        for path in log_path_list:
            merge_file = path + self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            if os.path.exists(merge_file):
                self.log_path_list.append(merge_file)
            else:
                print('Missing file from', path)

        # check if there are more than 1 to ensemble
        assert len(self.log_path_list) > 1, 'Cannot ensemble with 0 or 1 item'

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx']

        # always run test
        self.run_test = True

    def ensemble_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        output_file = output_folder + self.PREDICT_TEST_NAME
        if not os.path.exists(output_file):
            print('Copying template for output file')
            shutil.copy(self.log_path_list[0], output_folder)

        print('Creating ensemble results...')
        y_preds = []
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                y_preds.append(hf['predicted'][:])

        with h5py.File(output_file, 'a') as mf:
            mf['predicted'][:] = np.mean(y_preds, axis=0)
        print('Ensembled results saved to file')

        return self

    def concat_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        # first check the template
        with h5py.File(self.log_path_list[0], 'r') as f:
            ds_names = list(f.keys())
        ds = {name: [] for name in ds_names}

        # get the data
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                for key in ds:
                    ds[key].append(hf[key][:])

        # now merge them
        print('creating merged file')
        output_file = output_folder + self.PREDICT_TEST_NAME
        with h5py.File(output_file, 'w') as mf:
            for key, val in ds.items():
                mf.create_dataset(key, data=np.concatenate(val, axis=0))


@custom_architecture
class MultiInputModelLoaderV2(BaseModelLoader):
    def resize_by_axis(self, img, dim_1, dim_2, ax):
        resized_list = []
        # print(img.shape, ax, dim_1, dim_2)
        unstack_img_depth_list = tf.unstack(img, axis=ax)
        for j in unstack_img_depth_list:
            resized_list.append(
                image.resize(j, [dim_1, dim_2], method='bicubic'))
        stack_img = tf.stack(resized_list, axis=ax)
        # print(stack_img.shape)
        return stack_img

    def resize_along_dim(self, img, new_dim):
        dim_1, dim_2, dim_3 = new_dim

        resized_along_depth = self.resize_by_axis(img, dim_1, dim_2, 3)
        resized_along_width = self.resize_by_axis(
            resized_along_depth, dim_1, dim_3, 2)
        return resized_along_width

    def _create_dense_block(self, layer, connected_input):
        dense = layer['dense_block']
        if type(dense) == dict:
            layer_num = dense['layer_num']
        else:
            layer_num = dense

        dense_layers = [connected_input]
        final_concat = []
        for i in range(layer_num):
            next_tensor = layer_from_config(layer)
            if len(dense_layers) == 1:
                next_layer = next_tensor(connected_input)
            else:
                inp = concatenate(dense_layers[-2:])
                next_layer = next_tensor(inp)
                dense_layers.append(inp)

            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)
            dense_layers.append(next_layer)
            final_concat.append(next_layer)

        return concatenate(final_concat)

    def _create_res_block(self, layer, connected_input):
        res = layer['res_block']
        if type(res) == dict:
            layer_num = res['layer_num']
        else:
            layer_num = res
        next_layer = connected_input

        for i in range(layer_num):
            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)

            if 'activation' in layer['config']:
                activation = layer['config']['activation']
                del layer['config']['activation']

                next_layer = Activation(activation)(next_layer)

            next_layer = layer_from_config(layer)(next_layer)

        return Add()([connected_input, next_layer])

    def load(self):
        """
        Load the voxresnet neural network (2d and 3d)

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with vosresnet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        if type(self._input_params) == dict:
            self._input_params = [self._input_params]
        num_input = len(self._input_params)
        layers = [Input(**input_params) for input_params in self._input_params]
        saved_input = {f'input_{i}': layers[i] for i in range(num_input)}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                if len(layer['inputs']) > 1:
                    inputs = []
                    size_factors = None
                    for input_name in layer['inputs']:
                        # resize based on the first input
                        if size_factors:
                            if size_factors == saved_input[
                                    input_name].get_shape().as_list()[1:-1]:
                                next_input = saved_input[input_name]
                            else:
                                if len(size_factors) == 2:
                                    next_input = image.resize(
                                        saved_input[input_name],
                                        size_factors,
                                        # preserve_aspect_ratio=True,
                                        method='bilinear')
                                elif len(size_factors) == 3:

                                    next_input = self.resize_along_dim(
                                        saved_input[input_name],
                                        size_factors
                                    )

                                else:
                                    raise NotImplementedError(
                                        "Image shape is not supported ")
                            inputs.append(next_input)

                        else:  # set resize signal
                            inputs.append(saved_input[input_name])

                            # Based on the class_name, determine resize or not
                            # No resize is required for multi-input class.
                            # Example: Add, Multiple
                            # Concatenate and Addresize requires inputs
                            # # of the same shapes.
                            # Convolutional layers with multiple inputs
                            # # have hidden concatenation, so resize is also
                            # # required.
                            layer_class = layer['class_name']
                            if layer_class in resize_input_layers or \
                                    layer_class not in multi_input_layers:
                                size_factors = saved_input[
                                    input_name].get_shape().as_list()[1:-1]

                    # No concatenation for multi-input classes
                    if layer['class_name'] in multi_input_layers:
                        connected_input = inputs
                    else:
                        connected_input = concatenate(inputs)
                else:
                    connected_input = saved_input[layer['inputs'][0]]
            else:
                connected_input = layers[-1]

            # Resize back to original input
            if layer.get('resize_inputs'):
                size_factors = layers[0].get_shape().as_list()[1:-1]
                if size_factors != connected_input.get_shape().as_list()[1:-1]:
                    if len(size_factors) == 2:
                        connected_input = image.resize(
                            connected_input,
                            size_factors,
                            # preserve_aspect_ratio=True,
                            method='bilinear')
                    elif len(size_factors) == 3:
                        connected_input = self.resize_along_dim(
                            connected_input,
                            size_factors
                        )
                    else:
                        raise NotImplementedError(
                            "Image shape is not supported ")

            if 'res_block' in layer:
                next_layer = self._create_res_block(
                    layer, connected_input)
            elif 'dense_block' in layer:
                next_layer = self._create_dense_block(
                    layer, connected_input)
            else:
                next_tensor = layer_from_config(layer)

                next_layer = next_tensor(connected_input)

                if 'normalizer' in layer:
                    next_layer = layer_from_config(
                        layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[:num_input], outputs=layers[-1])


@custom_preprocessor
class ZScoreDensePreprocessor(BasePreprocessor):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, inputs, target=None):
        mean, std = self.mean, self.std
        if mean is None:
            mean = inputs.mean(axis=0)
            std = inputs.std(axis=0)
        else:
            mean = np.array(mean)
            std = np.array(std)
        std[std == 0] = 1

        return (inputs - mean)/std, target


@custom_datareader
class H5MultiReaderV2(DataReader):
    """DataReader that use data from an hdf5 file.

        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.

        Example:

        The dataset X contain 1000 samples, with 4 columns:
        x, y, z, t. Where x is the main input, y and z are supporting
        information (index, descriptions) and t is the target for
        prediction. We want to test 30% of this dataset, and have a
        cross validation of 100 samples.

        Then, the hdf5 containing dataset X should have 10 groups,
        each group contains 100 samples. We can name these groups
        'fold_1', 'fold_2', 'fold_3', ... , 'fold_9', 'fold_10'.
        Each group will then have 4 datasets: x, y, z and t, each of
        which has 100 items.

        Since x is the main input, then `x_name='x'`, and t is the
        target for prediction, then `y_name='t'`. We named the groups
        in the form of fold_n, then `fold_prefix='fold'`.

        Let's assume the data is stratified, we want to test on the
        last 30% of the data, so `test_folds=[8, 9, 10]`.
        100 samples is used for cross-validation. Thus, one option for
        `train_folds` and `val_folds` is `train_folds=[1,2,3,4,5,6]`
        and `val_folds=[7]`. Or in another experiment, you can set
        `train_folds=[2,3,4,5,6,7]` and `val_folds=[1]`.

        If the hdf5 didn't has any formular for group name, then you
        can set `fold_prefix=None` then put the full group name
        directly to `train_folds`, `val_folds` and `test_folds`.

        Parameters
        ----------
        filename : str
            The hdf5 file name that contains the data.
        batch_size : int, optional
            Number of sample to feeds in
            the neural network in each step, by default 32
        preprocessors : list of deoxys.data.Preprocessor, optional
            List of preprocessors to apply on the data, by default None
        x_name : str, optional
            Dataset name to be use as input, by default 'x'
        y_name : str, optional
            Dataset name to be use as target, by default 'y'
        batch_cache : int, optional
            Number of batches to be cached when reading the
            file, by default 10
        train_folds : list of int, or list of str, optional
            List of folds to be use as train data, by default None
        test_folds : list of int, or list of str, optional
            List of folds to be use as test data, by default None
        val_folds : list of int, or list of str, optional
            List of folds to be use as validation data, by default None
        fold_prefix : str, optional
            The prefix of the group name in the HDF5 file, by default 'fold'
        shuffle : bool, optional
            shuffle data while training, by default False
        augmentations : list of deoxys.data.Preprocessor, optional
            apply augmentation when generating traing data, by default None
    """

    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold', shuffle=False, augmentations=None,
                 other_input_names=None, other_preprocessors=None,
                 other_augmentations=None):
        """
        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.
        """
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5py.File(h5_filename, 'r')
        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.shuffle = shuffle

        self.preprocessors = preprocessors
        self.augmentations = augmentations

        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        self.other_input_names = other_input_names
        self.other_preprocessors = other_preprocessors
        self.other_augmentations = other_augmentations

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
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds, shuffle=self.shuffle,
            augmentations=self.augmentations,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors,
            other_augmentations=self.other_augmentations)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds, shuffle=False,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds, shuffle=False,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors)

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


class H5MultiDataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None,
                 shuffle=False, augmentations=None,
                 other_input_names=None, other_preprocessors=None,
                 other_augmentations=None):
        if not folds or not h5file:
            raise ValueError("h5file or folds is empty")

        # Checking for existence of folds and dataset
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
            if type(preprocessors) == list:
                for pp in preprocessors:
                    if not callable(getattr(pp, 'transform', None)):
                        raise ValueError(
                            'Preprocessor should have a "transform" method')
            else:
                if not callable(getattr(preprocessors, 'transform', None)):
                    raise ValueError(
                        'Preprocessor should have a "transform" method')

        if augmentations:
            if type(augmentations) == list:
                for pp in augmentations:
                    if not callable(getattr(pp, 'transform', None)):
                        raise ValueError(
                            'Augmentation must be a preprocessor with'
                            ' a "transform" method')
            else:
                if not callable(getattr(augmentations, 'transform', None)):
                    raise ValueError(
                        'Augmentation must be a preprocessor with'
                        ' a "transform" method')

        self.hf = h5file
        self.batch_size = batch_size
        self.seg_size = batch_size * batch_cache
        self.preprocessors = preprocessors
        self.augmentations = augmentations

        self.x_name = x_name
        self.y_name = y_name

        self.other_input_names = other_input_names
        self.other_preprocessors = other_preprocessors
        self.other_augmentations = other_augmentations

        self.shuffle = shuffle

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
        seg_num = np.ceil(
            h5file[self.folds[0]][y_name].shape[0] / self.seg_size)

        self.seg_list = np.arange(seg_num).astype(int)
        if self.shuffle:
            np.random.shuffle(self.seg_list)

    @property
    def description(self):
        if self.shuffle:
            raise Warning('The data is shuffled, the description results '
                          'may not accurate')
        if self._description is None:
            fold_names = self.folds
            description = []
            # find the shape of the inputs in the first fold
            shape = self.hf[fold_names[0]][self.x_name].shape
            obj = {'shape': shape[1:], 'total': shape[0]}

            for fold_name in fold_names[1:]:  # iterate through each fold
                shape = self.hf[fold_name][self.x_name].shape
                # if the shape are the same, increase the total number
                if np.all(obj['shape'] == shape[1:]):
                    obj['total'] += shape[0]
                # else create a new item
                else:
                    description.append(obj.copy())
                    obj = {'shape': shape[1:], 'total': shape[0]}

            # append the last item
            description.append(obj.copy())

            self._description = description
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
        if self._total_batch is None:
            total_batch = 0
            fold_names = self.folds

            for fold_name in fold_names:
                total_batch += np.ceil(
                    len(self.hf[fold_name][self.y_name]) / self.batch_size)
            self._total_batch = int(total_batch)
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
            seg_num = np.ceil(
                self.hf[cur_fold][self.y_name].shape[0] / self.seg_size)

            self.seg_list = np.arange(seg_num).astype(int)

            if self.shuffle:
                np.random.shuffle(self.seg_list)

        cur_fold = self.folds[self.fold_idx]
        cur_seg_idx = self.seg_list[self.seg_idx]

        start, end = cur_seg_idx * \
            self.seg_size, (cur_seg_idx + 1) * self.seg_size

        # print(cur_fold, cur_seg_idx, start, end)

        seg_x = self.hf[cur_fold][self.x_name][start: end]
        seg_y = self.hf[cur_fold][self.y_name][start: end]

        seg_others = [
            self.hf[cur_fold][name][start: end]
            for name in self.other_input_names
        ]

        return_indice = np.arange(len(seg_y))

        if self.shuffle:
            np.random.shuffle(return_indice)

        # Apply preprocessor
        if self.preprocessors:
            if type(self.preprocessors) == list:
                for preprocessor in self.preprocessors:
                    seg_x, seg_y = preprocessor.transform(
                        seg_x, seg_y)
            else:
                seg_x, seg_y = self.preprocessors.transform(
                    seg_x, seg_y)
        # Apply augmentation:
        if self.augmentations:
            if type(self.augmentations) == list:
                for preprocessor in self.augmentations:
                    seg_x, seg_y = preprocessor.transform(
                        seg_x, seg_y)
            else:
                seg_x, seg_y = self.augmentations.transform(
                    seg_x, seg_y)

        if self.other_preprocessors:
            for i, preprocessors in enumerate(self.other_preprocessors):
                for preprocessor in preprocessors:
                    seg_others[i], _ = preprocessor.transform(
                        seg_others[i], None)

        if self.other_augmentations:
            for i, preprocessors in enumerate(self.other_augmentations):
                for preprocessor in preprocessors:
                    seg_others[i], _ = preprocessor.transform(
                        seg_others[i], None)

        # increase seg index
        self.seg_idx += 1

        seg_others = [data[return_indice] for data in seg_others]

        return seg_x[return_indice], seg_y[return_indice], seg_others

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
        """
        while True:
            seg_x, seg_y, seg_others = self.next_seg()

            seg_len = len(seg_y)

            for i in range(0, seg_len, self.batch_size):
                batch_x = seg_x[i:(i + self.batch_size)]
                batch_others = [data[i:(i + self.batch_size)]
                                for data in seg_others]
                batch_y = seg_y[i:(i + self.batch_size)]

                yield [batch_x, *batch_others], batch_y


@custom_preprocessor
class ChannelRepeater(BasePreprocessor):
    def __init__(self, channel=0):
        if '__iter__' not in dir(channel):
            self.channel = [channel]
        else:
            self.channel = channel

    def transform(self, images, targets):
        return np.concatenate([images, images[..., self.channel]], axis=-1), targets


@custom_preprocessor
class ElasticDeform(BasePreprocessor):
    def __init__(self, sigma=4, points=3):
        self.sigma = sigma
        self.points = points

    def transform(self, x, y):
        return deform_random_grid([x, y], axis=[(1, 2, 3), (1, 2, 3)],
                                  sigma=self.sigma, points=self.points)


# @custom_preprocessor
# class ClassImageAugmentation2D(ImageAugmentation2D):
#     def transform(self, images, targets):
#         """
#         Apply augmentation to a batch of images

#         Parameters
#         ----------
#         images : np.array
#             the image batch
#         targets : np.array, optional
#             the target batch, by default None

#         Returns
#         -------
#         np.array
#             the transformed images batch (and target)
#         """
#         images = self.augmentation_obj.transform(images)
#         return images, targets
