import matplotlib.pyplot as plt
from deoxys.customize import custom_layer
from deoxys.model import load_model
from deoxys.customize import custom_layer
from deoxys.model.model import model_from_full_config
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.keras.backend import dropout


@custom_layer
class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def f1_score(y_true, y_pred, beta=1):
    assert len(y_true) == len(y_pred), "Shape not match"
    eps = 1e-8
    size = len(y_true.shape)
    reduce_ax = tuple(range(1, size))

    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true, axis=reduce_ax)
    target_positive = np.sum(y_true, axis=reduce_ax)
    predicted_positive = np.sum(y_pred, axis=reduce_ax)

    fb_numerator = (1 + beta ** 2) * true_positive + eps
    fb_denominator = (
        (beta ** 2) * target_positive + predicted_positive + eps
    )

    return fb_numerator / fb_denominator


# path to selected model
model_file = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/2d_perf/2d_unet_32_D3_P25_aug_affine/model/model.{:03d}.h5'
model_num = 65
# path to dataset
# ../../full_dataset_single.h5

original_model = load_model(model_file.format(model_num))

# ../../full_dataset_single.h5 (type in)

# get some sample images from the val set
val_gen = original_model.data_reader.val_generator.generate()
images, targets = np.zeros((160, 191, 265, 2)), np.zeros((160, 191, 265, 1))
for i in range(10):
    images[i*16:i*16+16], targets[i*16:i*16+16] = next(val_gen)

preds = original_model.predict(images)
selected_index = [0, 36, 45, 99, 120]

# check the Dice scores
f1_score(targets[selected_index], preds[selected_index]).round(3)


# model with monte carlo dropout + weights original model
dropout_model = model_from_full_config(
    'config/2d_unet_32_D3_dropout_P25_aug_affine.json', weights_file=model_file.format(model_num))

# ../../full_dataset_single.h5

new_preds = dropout_model.predict(images)
# Check the Dice drop
f1_score(targets, new_preds)

# Example of guided backprop function
# (conv2d_14 is name of the prediction layer,
# check the name with `dropout_model._model.summary()`)
dropout_res = dropout_model.guided_backprop('conv2d_14', images)
original_res = original_model.guided_backprop('conv2d_14', images)


# get the res and plot the images
def plot(images, targets):
    # predict 20 times, then get the std
    uncertainty_res = np.stack([dropout_model.predict(images)
                                for i in range(20)], axis=-1).std(axis=-1)
    # run 20 times, then get the std.
    # Note that there were different results for CT and PET images
    dropout_res = np.stack([dropout_model.guided_backprop(
        'conv2d_14', images) for i in range(20)], axis=-1).std(axis=-1)

    # for scaling the color map
    vmin, vmax = uncertainty_res.min(), uncertainty_res.max()
    d_vmin, d_vmax = dropout_res[..., 0].min(), dropout_res[..., 0].max()
    p_vmin, p_vmax = dropout_res[..., 1].min(), dropout_res[..., 1].max()

    # guided backprop for original model
    original_res = original_model.guided_backprop('conv2d_14', images)
    preds = original_model.predict(images)
    dscs = f1_score(targets, preds)

    nrow = len(images)
    ncol = 6

    for i, (image, target, pred, uncertainty_map, drop_uncertain, guided) in enumerate(zip(images, targets, preds, uncertainty_res, dropout_res, original_res)):
        # plot ct with contour
        plt.subplot(nrow, ncol, i*ncol + 1)
        plt.axis('off')
        plt.imshow(image[..., 0], 'gray')
        plt.contour(target[..., 0], 1, levels=[0.5], colors='yellow')
        plt.contour(pred[..., 0], 1, levels=[0.5], colors='red')
        plt.title(f'DSC: {dscs[i]:.3f}')

        # uncertainty map
        plt.subplot(nrow, ncol, i*ncol + 2)
        plt.axis('off')
        plt.imshow(image[..., 0], 'gray')
        uncertainty_map_ = uncertainty_map.copy()
        uncertainty_map_[uncertainty_map_ <
                         np.median(uncertainty_map_)] = np.nan
        plt.imshow(uncertainty_map_[..., 0],
                   'Oranges', alpha=0.5, vmin=vmin, vmax=vmax)
        plt.title(
            f'{uncertainty_map.min():.2f}-{uncertainty_map.max():.2f} Avg~{uncertainty_map.mean():.2f}\n'
            f'{np.quantile(uncertainty_map, 0.1):.2f}-{np.quantile(uncertainty_map, 0.25):.2f}-{np.quantile(uncertainty_map, 0.5):.2f}-{np.quantile(uncertainty_map, 0.75):.2f}-{np.quantile(uncertainty_map, 0.9):.2f}'
        )

        # uncertainty ct
        plt.subplot(nrow, ncol, i*ncol + 3)
        plt.axis('off')
        plt.imshow(image[..., 0], 'gray')
        drop_uncertain_c = drop_uncertain[..., 0].copy()
        drop_uncertain_c[drop_uncertain_c <
                         np.median(drop_uncertain_c)] = np.nan
        plt.imshow(drop_uncertain_c, 'Oranges',
                   alpha=0.5, vmin=d_vmin, vmax=d_vmax)
        plt.title(
            f'{drop_uncertain[..., 0].min():.2f}-{drop_uncertain[..., 0].max():.2f} Avg~{drop_uncertain[..., 0].mean():.2f}\n'
            f'{np.quantile(drop_uncertain[..., 0], 0.1):.2f}-{np.quantile(drop_uncertain[..., 0], 0.25):.2f}-{np.quantile(drop_uncertain[..., 0], 0.5):.2f}-{np.quantile(drop_uncertain[..., 0], 0.75):.2f}-{np.quantile(drop_uncertain[..., 0], 0.9):.2f}')

        # guided ct
        plt.subplot(nrow, ncol, i*ncol + 4)
        plt.axis('off')
        plt.imshow(image[..., 0], 'gray')
        plt.imshow(guided[..., 0], 'jet', alpha=0.5)

        # uncertainty pet
        plt.subplot(nrow, ncol, i*ncol + 5)
        plt.axis('off')
        plt.imshow(image[..., 1], 'gray')
        drop_uncertain_p = drop_uncertain[..., 1].copy()
        drop_uncertain_p[drop_uncertain_p <
                         np.median(drop_uncertain_p)] = np.nan
        plt.imshow(drop_uncertain_p, 'Oranges',
                   alpha=0.5, vmin=p_vmin, vmax=p_vmax)
        plt.title(
            f'{drop_uncertain[..., 1].min():.2f}-{drop_uncertain[..., 1].max():.2f} Avg~{drop_uncertain[..., 1].mean():.2f}\n'
            f'{np.quantile(drop_uncertain[..., 1], 0.1):.2f}-{np.quantile(drop_uncertain[..., 1], 0.25):.2f}-{np.quantile(drop_uncertain[..., 1], 0.5):.2f}-{np.quantile(drop_uncertain[..., 1], 0.75):.2f}-{np.quantile(drop_uncertain[..., 1], 0.9):.2f}')

        # guided pet
        plt.subplot(nrow, ncol, i*ncol + 6)
        plt.axis('off')
        plt.imshow(image[..., 1], 'gray')
        plt.imshow(guided[..., 1], 'jet', alpha=0.5)

    return drop_uncertain, original_res, uncertainty_res


dropout_res, original_res, uncertainty_res = plot(
    images[selected_index], targets[selected_index])
plt.tight_layout()
plt.show()
