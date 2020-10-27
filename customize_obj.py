from deoxys.customize import custom_architecture, custom_preprocessor, \
    custom_datareader
from deoxys.loaders.architecture import Vnet
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator

from deoxys.keras.models import Model as KerasModel
from deoxys.keras.layers import Input, concatenate, \
    Add, Activation

from tensorflow import image
import tensorflow as tf

from deoxys.model.layers import layer_from_config

import numpy as np

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
        self.vmin = vmin
        self.vmax = vmax

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
