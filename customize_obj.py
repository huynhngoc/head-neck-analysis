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
                height = (height // divide_factor + 1) * divide_factor

            if width % divide_factor != 0:
                width = (width // divide_factor + 1) * divide_factor

            images = image.resize_with_pad(images, height, width)
            targets = image.resize_with_pad(targets, height, width)
            return images, targets

        if len(shape) == 3:
            height, width, z = shape

            if height % divide_factor != 0:
                new_height = (height // divide_factor + 1) * divide_factor
            else:
                new_height = height

            if width % divide_factor != 0:
                new_width = (width // divide_factor + 1) * divide_factor
            else:
                new_width = WindowsError

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
