import gc
from itertools import product
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
from deoxys.utils import file_finder, load_json_config
from deoxys.customize import custom_architecture, custom_datareader, custom_layer
from deoxys.loaders import load_data
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator
from deoxys.model.layers import layer_from_config
import tensorflow_addons as tfa
from deoxys.model.losses import Loss, loss_from_config
from deoxys.customize import custom_loss, custom_preprocessor
from deoxys.data import ImageAugmentation2D


multi_input_layers = ['Add', 'AddResize', 'Concatenate', 'Multiply']
resize_input_layers = ['Concatenate', 'AddResize']


@custom_layer
class InstanceNormalization(tfa.layers.InstanceNormalization):
    pass


@custom_layer
class AddResize(Add):
    pass


@custom_loss
class BinaryMacroFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_macro_fbeta", beta=1):
        super().__init__(reduction, name)

        self.beta = beta

    def call(self, target, prediction):
        eps = 1e-8
        target = tf.cast(target, prediction.dtype)

        true_positive = tf.math.reduce_sum(prediction * target)
        target_positive = tf.math.reduce_sum(tf.math.square(target))
        predicted_positive = tf.math.reduce_sum(
            tf.math.square(prediction))

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


@custom_architecture
class MultiInputModelLoader(BaseModelLoader):
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
                connected_input = layers[i]

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
