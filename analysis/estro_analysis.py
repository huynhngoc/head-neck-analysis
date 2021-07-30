import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from medvis import apply_cmap_with_blend

result_file = 'csv/estro_results_val.csv'
df = pd.read_csv(result_file)


def Q1(x):
    return x.quantile(0.25)


def Q3(x):
    return x.quantile(0.75)


df.groupby('model').agg(
    {'val': [Q1, 'median', Q3, 'mean', 'std']})


result_file = 'csv/estro_results_test.csv'
df = pd.read_csv(result_file)
df.groupby('model').agg(
    {'test': [Q1, 'median', Q3, 'mean', 'std']})

path = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/2d_perf/2d_unet_32_norm_aug'
path2 = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/2d_perf/2d_unet_P10_aug'
with h5py.File(path + '/test/prediction_test.h5', 'r') as f:
    # images = f['x']['91'][:]
    targets = f['y']['91'][:]
    predicts = f['predicted']['91'][:]

with h5py.File(path2 + '/test/prediction_test.h5', 'r') as f:
    images = f['x']['91'][:]
with h5py.File('../../hn_perf/3d_unet_32_norm_aug' + '/test/prediction_test.h5', 'r') as f:
    print(f['x'].keys())
    predicts_2 = f['predicted']['91'][:]
    images_2 = f['x']['91'][:]


predict_ = (predicts > 0.5).astype(float)
predict2_ = (predicts_2 > 0.5).astype(float)
# plt.imshow(predict2_[60][..., 0]);plt.show()
plt.subplot(1, 2, 1)
plt.imshow(images_2[102][..., 0])
plt.contour(predict2_[102][..., 0])
plt.subplot(1, 2, 2)
plt.imshow(images[68][..., 0])
plt.show()

for i in range(173):
    if np.allclose(images[68][..., 0], images_2[i][..., 0]):
        print(i)
        print(np.all(images[68][..., 0] == images_2[i][..., 0]))


def plot_pet_blend_ct_image(ax, img, contour, pred, pred2):
    ax.imshow(img[..., 0], 'gray', vmin=0.25, vmax=0.75)
    ax.imshow(apply_cmap_with_blend(
        img[..., 1], 'inferno', vmin=0, vmax=1))
    ax.axis('off')
    true_con = ax.contour(contour[..., 0], 1, levels=[0.5], colors='#f8f1a9')
    pred_con = ax.contour(pred[..., 0], 1, levels=[0.5], colors='#eaa383')
    pred2_con = ax.contour(pred2[..., 0], 1, levels=[0.5], colors='#00ffff')

    ax.legend([true_con.collections[0],
               pred_con.collections[0],
               pred2_con.collections[0]], ['Ground Truth', 'Best 2D model', 'Best 3D model'],
              loc='center', bbox_to_anchor=(0.5, -0.05),
              ncol=3,
              #    fontsize='small'
              )


sns.set_style('white')
i = 68
img, contour, pred, pred2 = images[i], targets[i], predict_[i], predict2_[102]
# ax = plt.subplot(2, 1, 1)
plot_pet_blend_ct_image(plt, img, contour, pred, pred2)

plt.show()
