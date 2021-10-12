import gc
from itertools import product
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
import h5py
import numpy as np
import tensorflow as tf
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.experiment import Experiment
from deoxys.utils import file_finder, load_json_config
from deoxys.customize import custom_datareader, custom_layer
from deoxys.loaders import load_data
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator
import tensorflow_addons as tfa
from tensorflow.keras.layers import Add
from deoxys.model.losses import Loss, loss_from_config
from deoxys.customize import custom_loss, custom_preprocessor
from deoxys.data import ImageAugmentation2D


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
