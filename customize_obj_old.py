from deoxys.customize import custom_architecture, custom_preprocessor, \
    custom_datareader
from deoxys.loaders.architecture import Vnet
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator

from deoxys.keras.models import Model as KerasModel
from deoxys.keras.layers import Input, concatenate, \
    Add, Activation

from tensorflow import image
import tensorflow as tf

from deoxys.model.layers import layer_from_config

import numpy as np

import h5py
from deoxys.utils import file_finder

multi_input_layers = ['Add', 'Concatenate']


@custom_preprocessor
class Padding(BasePreprocessor):
    def __init__(self, depth=4, mode='auto'):
        self.depth = depth
        self.mode = mode

    def transform(self, images, targets):
        image_shape = images.shape
        target_shape = targets.shape

        shape = image_shape[1:-1]

        divide_factor = 2 ** self.depth

        if len(shape) == 2:
            height, width = shape
            if height % divide_factor != 0:
                new_height = (height // divide_factor + 1) * divide_factor

            if width % divide_factor != 0:
                new_width = (width // divide_factor + 1) * divide_factor

            # images = image.resize_with_pad(images, height, width)
            # targets = image.resize_with_pad(targets, height, width)

            # return images, targets

            new_images = np.zeros(
                (image_shape[0],
                 new_height, new_width,
                 image_shape[-1]))
            new_targets = np.zeros(
                (target_shape[0],
                    new_height, new_width,
                    target_shape[-1]))

            min_h = (new_height - height) // 2
            min_w = (new_width - width) // 2

            new_images[:, min_h: min_h+height,
                       min_w: min_w+width, :] = images

            new_targets[:, min_h: min_h+height,
                        min_w: min_w+width, :] = targets

            return new_images, new_targets

        if len(shape) == 3:
            height, width, z = shape

            if height % divide_factor != 0:
                new_height = (height // divide_factor + 1) * divide_factor
            else:
                new_height = height

            if width % divide_factor != 0:
                new_width = (width // divide_factor + 1) * divide_factor
            else:
                new_width = width

            if z % divide_factor != 0:
                new_z = (z // divide_factor + 1) * divide_factor
            else:
                new_z = z

            if self.mode == 'edge':
                pass
            else:  # default - pad with zeros
                new_images = np.zeros(
                    (image_shape[0],
                     new_height, new_width, new_z,
                     image_shape[-1]))
                new_targets = np.zeros(
                    (target_shape[0],
                     new_height, new_width, new_z,
                     target_shape[-1]))

            min_h = (new_height - height) // 2
            min_w = (new_width - width) // 2
            min_z = (new_z - z) // 2

            new_images[:, min_h: min_h+height,
                       min_w: min_w+width, min_z:min_z+z, :] = images

            new_targets[:, min_h: min_h+height,
                        min_w: min_w+width, min_z:min_z+z, :] = targets

            return new_images, new_targets

        raise RuntimeError('Does not support 4D tensors')


@custom_preprocessor
class ImageNormalizer(BasePreprocessor):
    def __init__(self, vmin=0, vmax=255):
        """
        Normalize all channels to the range 0-1

        Args:
            vmin (int, or list,  optional): [description]. Defaults to 0.
            vmax (int, or list optional): [description]. Defaults to 255.
        """
        self.vmin = np.array(vmin) if type(vmin) == list else vmin
        self.vmax = np.array(vmax) if type(vmax) == list else vmax

    def transform(self, images, targets):
        transformed_images = (np.array(images) - self.vmin) / \
            (self.vmax - self.vmin)
        transformed_images = transformed_images.clip(0, 1)

        return transformed_images, targets


@custom_architecture
class VnetModified(Vnet):
    def resize_by_axis(self, img, dim_1, dim_2, ax):
        resized_list = []
        # print(img.shape, ax, dim_1, dim_2)
        unstack_img_depth_list = tf.unstack(img, axis=ax)
        # method = None
        for im in unstack_img_depth_list:
            # if not method:
            #     if im.shape[1] > dim_1:
            #         method = 'bilinear'
            #     else:
            #         method = 'bicubic'
            #     print(method)

            resized_list.append(
                image.resize(im, [dim_1, dim_2], method='bilinear'))
        stack_img = tf.stack(resized_list, axis=ax)
        # print(stack_img.shape)
        return stack_img


@custom_architecture
class DenseNetV2(Vnet):
    def load(self):
        """
        Load the unet neural network. Use Conv3d

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with unet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
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

                    else:
                        inputs.append(saved_input[input_name])
                        size_factors = saved_input[
                            input_name].get_shape().as_list()[1:-1]

                connected_input = concatenate(inputs)
            else:
                connected_input = layers[i]

            if 'dense_block' in layer:
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

        return KerasModel(inputs=layers[0], outputs=layers[-1])

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

    def _create_dense_blockv1(self, layer, connected_input, input):
        dense = layer['dense_block']
        if type(dense) == dict:
            layer_num = dense['layer_num']
        else:
            layer_num = dense

        dense_layers = [connected_input]
        for i in range(layer_num):
            next_tensor = layer_from_config(layer)
            if len(dense_layers) == 1:
                next_layer = next_tensor(connected_input)
            else:
                inp = concatenate(dense_layers)
                next_layer = next_tensor(inp)
                print(inp)
                KerasModel(inputs=input, outputs=inp).summary()
                KerasModel(inputs=input, outputs=next_layer).summary()

            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)
            dense_layers.append(next_layer)

        return concatenate(dense_layers[1:])


@custom_datareader
class HDF5ReaderV2(HDF5Reader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold', patch_size=48, overlap=8):
        super().__init__(filename, batch_size, preprocessors,
                         x_name, y_name, batch_cache,
                         train_folds, test_folds, val_folds,
                         fold_prefix)

        self.patch_size = patch_size
        self.overlap = overlap

    @property
    def train_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return HDF5DataGeneratorV2(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds, is_training=True,
            patch_size=self.patch_size, overlap=self.overlap)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return HDF5DataGeneratorV2(
            self.hf, batch_size=1 if self.patch_size else self.batch_size,
            batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return HDF5DataGeneratorV2(
            self.hf, batch_size=1 if self.patch_size else self.batch_size,
            batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds)


class HDF5DataGeneratorV2(HDF5DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None, is_training=False,
                 patch_size=None, overlap=8):
        if is_training and patch_size:
            self.batch_size_patch = batch_size
            batch_size = 1
            batch_cache = 1

        self._z_axis = None

        super().__init__(h5file, batch_size, batch_cache,
                         preprocessors,
                         x_name, y_name, folds)
        self.is_training = is_training
        self.patch_size = patch_size
        self.overlap = overlap

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
        """
        while True:
            # When all batches of data are yielded, move to next seg
            if self.index >= self.seg_size or \
                    self.seg_index + self.index >= self.fold_len:
                self.next_seg()

            # Index may has been reset. Thus, call after next_seg
            index = self.index

            # The last batch of data may not have less than batch_size items
            if index + self.batch_size >= self.seg_size or \
                    self.seg_index + index + self.batch_size >= self.fold_len:
                batch_x = self.x_cur[index:]
                batch_y = self.y_cur[index:]
            else:
                # Take the next batch
                batch_x = self.x_cur[index:(index + self.batch_size)]
                batch_y = self.y_cur[index:(index + self.batch_size)]

            self.index += self.batch_size

            if self.is_training and self.patch_size:
                im, label = [], []
                for i in range(0, self.z_axis - self.patch_size,
                               self.overlap):
                    im.append(batch_x[0][i: i + self.patch_size])
                    label.append(batch_y[0][i: i + self.patch_size])

                    if len(im) == self.batch_size_patch:
                        yield np.array(im), np.array(label)
                        im, label = [], []

                if len(im) > 0:
                    yield np.array(im), np.array(label)

            else:
                yield batch_x, batch_y

    @property
    def total_batch(self):
        if self.is_training and self.patch_size:
            total_items = super().total_batch

            return total_items * np.ceil(
                (self.z_axis - self.patch_size) / self.overlap)

        else:
            return super().total_batch

    @property
    def z_axis(self):
        if self._z_axis is None:
            x = self.hf[self.folds[0]][self.x_name][:1]
            y = self.hf[self.folds[0]][self.y_name][:1]

            if self.preprocessors:
                if type(self.preprocessors) == list:
                    for preprocessor in self.preprocessors:
                        x, y = preprocessor.transform(
                            x, y)
                else:
                    x, y = self.preprocessors.transform(x, y)

            self._z_axis = x.shape[1]

        return self._z_axis


@custom_architecture
class VoxResNet(Vnet):
    def load(self):
        """
        Load the unet neural network. Use Conv3d

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with unet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                if len(layer['inputs']) > 1:
                    inputs = []
                    size_factors = None
                    for input_name in layer['inputs']:
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

                        else:
                            inputs.append(saved_input[input_name])
                            size_factors = saved_input[
                                input_name].get_shape().as_list()[1:-1]

                    if layer['class_name'] in multi_input_layers:
                        connected_input = inputs
                    else:
                        connected_input = concatenate(inputs)
                else:
                    connected_input = saved_input[layer['inputs'][0]]
            else:
                connected_input = layers[i]

            if 'res_block' in layer:
                next_layer = self._create_res_block(
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

        return KerasModel(inputs=layers[0], outputs=layers[-1])

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


@custom_datareader
class H5Reader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold', patch_size=48, overlap=8, shuffle=False,
                 stratify=False, postprocessor=None):
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
        self.patch_size = patch_size
        self.overlap = overlap

        self.shuffle = shuffle
        self.stratify = stratify

        self.preprocessors = preprocessors
        self.postprocessor = postprocessor

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
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds, shuffle=self.shuffle)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds, shuffle=False)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds, shuffle=False)

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


class H5DataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None, shuffle=False):
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

        self.hf = h5file
        self.batch_size = batch_size
        self.seg_size = batch_size * batch_cache
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name

        self.shuffle = shuffle

        self.folds = str_folds

        self._total_batch = None

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

        # increase seg index
        self.seg_idx += 1

        return seg_x[return_indice], seg_y[return_indice]

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


class H5MergePatches:
    def __init__(self, ref_file, predicted_file,
                 map_column, merge_file, save_file,
                 patch_size, overlap,
                 folds, fold_prefix='fold',
                 original_input_dataset='x',
                 original_target_dataset='y',
                 predicted_dataset='predicted', target_dataset='y',
                 input_dataset='x'
                 ):

        # def __init__(self, ref_file, map_file, map_column, merge_file, save_file,
        #              predicted_dataset='predicted', target_dataset='y',
        #              input_dataset='x'):
        self.ref_file = ref_file
        self.predicted_file = predicted_file
        self.map_column = map_column
        self.merge_file = merge_file
        self.save_file = save_file

        self.ref_inputs = original_input_dataset
        self.ref_targets = original_target_dataset

        self.predicted = predicted_dataset
        self.target = target_dataset
        self.inputs = input_dataset

        if fold_prefix:
            self.folds = ['{}_{}'.format(
                fold_prefix, fold) for fold in folds]
        else:
            self.folds = folds

        self.patch_size = patch_size
        self.overlap = overlap

        print('merging images of patch', patch_size)

    def _save_inputs_target_to_merge_file(self, fold, meta, index):
        with h5py.File(self.ref_file, 'r') as f:
            inputs = f[fold][self.ref_inputs][index]
            targets = f[fold][self.ref_targets][index]

        print(inputs.shape, self.inputs, meta)

        with h5py.File(self.merge_file, 'a') as mf:
            mf[self.inputs].create_dataset(
                meta, data=inputs, compression="gzip")
            mf[self.target].create_dataset(
                meta, data=targets, compression="gzip")

    def _merge_patches_to_merge_file(self, meta, start_cursor):
        with h5py.File(self.merge_file, 'r') as mf:
            shape = mf[self.target][meta].shape[:-1]

        # fix patch size
        if '__iter__' not in dir(self.patch_size):
            self.patch_size = [self.patch_size] * len(shape)

        indice = get_patch_indice(shape, self.patch_size, self.overlap)
        next_cursor = start_cursor + len(indice)

        with h5py.File(self.predicted_file, 'r') as f:
            data = f[self.predicted][start_cursor: next_cursor]

        predicted = np.zeros(shape)
        weight = np.zeros(shape)

        for i in range(len(indice)):
            x, y, z = indice[i]
            w, h, d = self.patch_size
            predicted[x:x+w, y:y+h, z:z+d] = predicted[x:x+w, y:y+h, z:z+d] \
                + data[i][..., 0]
            weight[x:x+w, y:y+h, z:z+d] = weight[x:x+w, y:y+h, z:z+d] \
                + np.ones(self.patch_size)

        predicted = (predicted/weight)[..., np.newaxis]

        with h5py.File(self.merge_file, 'a') as mf:
            mf[self.predicted].create_dataset(
                meta, data=predicted, compression="gzip")

        return next_cursor

    def post_process(self):
        # create merge file
        with h5py.File(self.merge_file, 'w') as mf:
            mf.create_group(self.inputs)
            mf.create_group(self.target)
            mf.create_group(self.predicted)

        data = []
        start_cursor = 0
        for fold in self.folds:
            with h5py.File(self.ref_file, 'r') as f:
                meta_data = f[fold][self.map_column][:]
                data.extend(meta_data)
                for index, meta in enumerate(meta_data):
                    self._save_inputs_target_to_merge_file(
                        fold, str(meta), index)
                    start_cursor = self._merge_patches_to_merge_file(
                        str(meta), start_cursor)

        # create map file
        df = pd.DataFrame(data, columns=[self.map_column])
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
    TEST_MAP_NAME = '/result.csv'

    def __init__(self, log_base_path='logs',
                 temp_base_path='',
                 analysis_base_path='',
                 map_meta_data=None, main_meta_data='',
                 run_test=False):
        self.temp_base_path = temp_base_path
        self.log_base_path = log_base_path
        self.analysis_base_path = analysis_base_path or log_base_path

        if not os.path.exists(self.analysis_base_path):
            os.mkdir(self.analysis_base_path)

        if not os.path.exists(self.analysis_base_path + self.PREDICTION_PATH):
            os.mkdir(self.analysis_base_path + self.PREDICTION_PATH)

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

    def calculate_fscore_single_3d(self):
        self.calculate_fscore_single()
        if not self.run_test:
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH

            main_log_folder = self.log_base_path + self.MAP_PATH
            os.rename(map_folder, main_log_folder)
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME

            main_result_file_name = test_folder + self.TEST_MAP_NAME

            os.rename(map_filename, main_result_file_name)

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

    def merge_3d_patches(self):
        print('merge 3d patches to 3d images')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            # map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            # map_filename = map_folder + self.SINGLE_MAP_NAME

            merge_path = self.analysis_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH

            if not os.path.exists(main_log_folder):
                os.makedirs(main_log_folder)
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5MergePatches(
                    ref_file=self.dataset_filename,
                    predicted_file=predicted_path.format(epoch=epoch),
                    map_column=self.main_meta_data,
                    merge_file=merge_path.format(epoch=epoch),
                    save_file=main_log_filename.format(epoch=epoch),
                    patch_size=self.data_reader.patch_size,
                    overlap=self.data_reader.overlap,
                    folds=self.data_reader.val_folds,
                    fold_prefix='',
                    original_input_dataset=self.data_reader.x_name,
                    original_target_dataset=self.data_reader.y_name,
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            H5MergePatches(
                ref_file=self.dataset_filename,
                predicted_file=predicted_path,
                map_column=self.main_meta_data,
                merge_file=merge_path,
                save_file=main_result_file_name,
                patch_size=self.data_reader.patch_size,
                overlap=self.data_reader.overlap,
                folds=self.data_reader.val_folds,
                fold_prefix='',
                original_input_dataset=self.data_reader.x_name,
                original_target_dataset=self.data_reader.y_name,
            ).post_process()

        return self

    def calculate_fscore(self):
        print('calculating dice score per 3d image')
        if not self.run_test:
            merge_path = self.analysis_base_path + \
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
                main_result_file_name,
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

        res_df[monitor] = results
        best_epoch = epochs[res_df[monitor].argmax()]

        res_df.to_csv(self.log_base_path + '/log_new.csv', index=False)

        if keep_best_only:
            for epoch in epochs:
                if epoch != best_epoch:
                    os.remove(self.analysis_base_path + self.PREDICTION_PATH +
                              self.PREDICTION_NAME.format(epoch=epoch))
                elif self.log_base_path != self.analysis_base_path:
                    # move the best prediction to main folder
                    shutil.copy(self.analysis_base_path + self.PREDICTION_PATH +
                                self.PREDICTION_NAME.format(epoch=epoch),
                                self.log_base_path + self.PREDICTION_PATH +
                                self.PREDICTION_NAME.format(epoch=epoch))

                    os.remove(self.analysis_base_path + self.PREDICTION_PATH +
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
