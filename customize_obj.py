from deoxys.customize import custom_architecture
from deoxys.loaders.architecture import BaseModelLoader, Vnet

from deoxys.keras.models import Model as KerasModel
from deoxys.keras.layers import Input, concatenate, Lambda
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
