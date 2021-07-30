import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

result_file = 'csv/2d_test_results.csv'
df = pd.read_csv(result_file)


def Q1(x):
    return x.quantile(0.25)


def Q3(x):
    return x.quantile(0.75)


df.groupby('model').agg(
    {'f1_score': ['min', Q1, 'median', Q3, 'max', 'mean', 'std']})
orders = [
    '2d_unet', '2d_unet_48_norm',
    '2d_unet_64_D3_P10_aug_affine',
    '2d_unet_P10_aug',
    '2d_unet_48_P25_aug_affine',
    '2d_unet_48_P10_aug_affine',
    '2d_unet_48_norm_aug',
    '2d_unet_norm_aug', '2d_unet_32_norm_aug'
]

plt.subplot(1, 2, 1)
sns.violinplot(x=df['model'],
               y=df['f1_score'],
               order=orders,
               #    hue=df['augmentation']
               cut=0,
               inner='quartiles'
               )
# plt.title('Dice score on different normalization')

plt.subplot(1, 2, 2)
sns.violinplot(x=df['model'],
               y=df['f1_score'],
               order=orders,
               #    hue=df['augmentation']
               cut=0,
               inner='box'
               )
# plt.title('Dice score on different normalization')

plt.show()
