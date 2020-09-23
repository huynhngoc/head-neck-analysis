from deoxys.customize import custom_architecture, custom_preprocessor, \
    custom_datareader
from deoxys.loaders.architecture import BaseModelLoader, Vnet
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator

from deoxys.keras.models import Model as KerasModel
from deoxys.keras.layers import Input, concatenate, Lambda, Concatenate
from deoxys.utils import is_keras_standalone
from tensorflow import image
import tensorflow as tf

from deoxys.model.layers import layer_from_config
from deoxys.utils import deep_copy

import numpy as np


@custom_preprocessor
class Padding(BasePreprocessor):
    def __init__(self, depth=4):
        self.depth = depth

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

            new_images = np.zeros(
                (image_shape[0], new_height, new_width, new_z, image_shape[-1]))
            new_targets = np.zeros(
                (target_shape[0], new_height, new_width, new_z, target_shape[-1]))

            min_h = (new_height - height) // 2
            min_w = (new_width - width) // 2
            min_z = (new_z - z) // 2

            new_images[:, min_h: min_h+height,
                       min_w: min_w+width, min_z:min_z+z, :] = images

            new_targets[:, min_h: min_h+height,
                        min_w: min_w+width, min_z:min_z+z, :] = targets

            return new_images, new_targets

        raise RuntimeError('Does not support 4D tensors')


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
            folds=self.train_folds, is_training=True)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return HDF5DataGeneratorV2(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
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
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds)


class HDF5DataGeneratorV2(HDF5DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None, is_training=False):
        super().__init__(h5file, batch_size, batch_cache,
                         preprocessors,
                         x_name, y_name, folds)
        self.is_training = is_training

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

            if self.batch_size == 1 and self.is_training:
                for i in range(11):
                    im = batch_x[0][i*16: i*16 + 16]
                    label = batch_y[0][i*16: i*16 + 16]
                    print(np.array([im]).shape)
                    yield np.array([im]), np.array([label])

                for i in range(10):
                    im = [batch_x[0][8 + i*16: i*16 + 24]]
                    label = [batch_y[0][8 + i*16: i*16 + 24]]
                    yield im, label

            else:
                yield batch_x, batch_y

    @property
    def total_batch(self):
        if self.is_training:
            return super().total_batch * 21
        else:
            return super().total_batch


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
