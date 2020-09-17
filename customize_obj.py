from deoxys.customize import custom_architecture
from deoxys.loaders.architecture import BaseModelLoader, Vnet

from deoxys.keras.models import Model as KerasModel
from deoxys.keras.layers import Input, concatenate, Lambda, Concatenate
from deoxys.utils import is_keras_standalone
from tensorflow import image
import tensorflow as tf

from deoxys.model.layers import layer_from_config
from deoxys.utils import deep_copy


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
