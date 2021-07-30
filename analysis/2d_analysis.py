from scipy.stats import friedmanchisquare
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

result_file = 'csv/all_res.csv'

df = pd.read_csv(result_file)

df.columns = [str(col) for col in df.columns]

filter_depth = df['filter'].astype(str) + '_' + df['depth'].astype(str)
df.insert(loc=len(df.columns) - 15, column='filter_depth', value=filter_depth)

filter_depth = df['filter'].astype(
    str) + ' filters\n' + df['depth'].astype(str) + ' up/down\nsampling layers'
df.insert(loc=len(df.columns) - 15,
          column='Model complexity', value=filter_depth)

new_aug_col = df['aug'].copy()
new_aug_col[df['aug'].isnull()] = 'No augmentation'
new_aug_col[df['aug'] == 'Affine'] = 'Affine transform'
new_aug_col[df['aug'] == 'Default'] = 'Affine transform &\nIntensity Changes'
df.insert(loc=len(df.columns) - 15, column='augmentation', value=new_aug_col)


sns.violinplot(x=df['depth'], y=df['dice'],
               hue=df['augmentation'], scale='count')
plt.title('Dice score on different normalization')
plt.show()

group_df = df.melt(id_vars=df.columns[:-15],
                   var_name='Patient idx', value_name='Dice')


def Q1(x):
    return x.quantile(0.25)


def Q3(x):
    return x.quantile(0.75)


def to_int(x):
    return x.astype(int)


group_df.groupby('Patient idx').agg(
    {'Dice': ['min', Q1, 'median', Q3, 'max', 'mean', 'std']}).sort_index(key=to_int)
group_df[group_df['normalize'] != 'FALSE'].groupby('name').agg({'Dice': 'mean', 'augmentation': 'min'}).groupby(
    'augmentation').agg({'Dice': ['min', Q1, 'median', Q3, 'max', 'mean', 'std']})

sns.set_theme(style='whitegrid')
sns.violinplot(x=group_df['depth'],
               y=group_df['Dice'],
               hue=group_df['augmentation'],
               hue_order=['No augmentation', 'Affine transform',
                          'Affine transform &\nIntensity Changes'],
               legend=False,
               #    inner='box',
               #    bw=0.1,
               cut=0
               )
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
           ncol=3, fontsize='small')
# plt.ylim([0.5, 0.9])
plt.show()

sns.violinplot(x=group_df['Patient idx'],
               y=group_df['Dice'],
               hue=group_df['normalize'],
               )
plt.show()

sns.violinplot(x=group_df[group_df['aug'].isnull()]['Patient idx'],
               y=group_df[group_df['aug'].isnull()]['Dice'],
               hue=group_df[group_df['aug'].isnull()]['normalize'],
               )
plt.show()

sns.violinplot(x=group_df[group_df['aug'].notnull()]['Patient idx'],
               y=group_df[group_df['aug'].notnull()]['Dice'],
               hue=group_df[group_df['aug'].notnull()]['normalize'],
               )
plt.show()


sns.violinplot(x=group_df['Patient idx'],
               y=group_df['Dice'],
               hue=group_df['filter'],
               )
plt.show()

sns.violinplot(x=group_df['Patient idx'],
               y=group_df['Dice'],
               hue=group_df['depth'],
               )
plt.show()

largest_index = df['90'].nlargest(15).index
df.columns[15]
selected_column = list(df.columns[:-15]) + list(df.columns[15:16])
df[df.index.isin(largest_index)][selected_column + ['177'] + ['38']]


sns.violinplot(x=group_df[group_df.aug.notnull()]['Patient idx'],
               y=group_df[group_df.aug.notnull()]['Dice'],
               hue=group_df[group_df.aug.notnull()]['filter'],
               cut=0
               )
plt.title('With Augmentation')
plt.show()


sns.violinplot(x=group_df[group_df.aug.isnull()]['Patient idx'],
               y=group_df[group_df.aug.isnull()]['Dice'],
               hue=group_df[group_df.aug.isnull()]['filter'],
               cut=0
               )
plt.title('No Augmentation')
plt.show()

indice = group_df[group_df['Patient idx'] == '90']['Dice'].nlargest(5).index
indice
group_df[group_df.index.isin(indice)]


indice = group_df[group_df['Patient idx'] == '177']['Dice'].nlargest(5).index
indice
group_df[group_df.index.isin(indice)]

indice = group_df[group_df['Patient idx'] == '38']['Dice'].nlargest(5).index
indice
group_df[group_df.index.isin(indice)]


sns.violinplot(x=group_df[~group_df['Patient idx'].isin(['90', '177'])]['augmentation'],
               y=group_df[~group_df['Patient idx'].isin(
                   ['90', '177'])]['Dice'],
               hue=group_df[~group_df['Patient idx'].isin(
                   ['90', '177'])]['normalize'],
               #    cut=0
               )
plt.title('')
plt.show()


sns.set_theme(style='whitegrid')
df = df.rename(columns={'dice': 'DSC'})
sns.violinplot(x=df['Model complexity'],
               y=df['DSC'],
               hue=df['augmentation'],
               cut=0,
               order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
                      for f in [32, 48, 64]],
               hue_order=['No augmentation', 'Affine transform',
                          'Affine transform &\nIntensity Changes'],
               legend=False,
               inner='quartile',
               scale='area'
               )
plt.legend(loc='lower right')
plt.show()
df = df.rename(columns={'DSC': 'dice'})

df.groupby(['augmentation', 'Model complexity']).agg(
    {'dice': ['min', Q1, 'median', Q3, 'max', 'mean', 'std', 'count']})

df


df = df.rename(columns={'dice': 'DSC'})
df = df.rename(columns={'filter': 'No. filters'})
df = df.rename(columns={'depth': 'Depth'})
plt.subplot(2, 2, 1)
sns.violinplot(x=df['No. filters'],
               y=df['DSC'],
               hue=df['augmentation'],
               cut=0,
               #    order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
               #           for f in [32, 48, 64]],
               hue_order=['No augmentation', 'Affine transform',
                          'Affine transform &\nIntensity Changes'],
               legend=False,
               #    inner='quartile',
               scale='area'
               )
plt.legend().set_visible(False)
plt.subplot(2, 2, 2)
ax = sns.violinplot(x=df['Depth'],
                    y=df['DSC'],
                    hue=df['augmentation'],
                    cut=0,
                    #    order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
                    #           for f in [32, 48, 64]],
                    hue_order=['No augmentation', 'Affine transform',
                               'Affine transform &\nIntensity Changes'],
                    legend=False,
                    #    inner='quartile',
                    scale='area',
                    squeeze=True
                    )
ax.set_ylabel('')
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
           ncol=3, fontsize='small')
plt.show()
df = df.rename(columns={'DSC': 'dice'})
df = df.rename(columns={'No. filters': 'filter'})
df = df.rename(columns={'Depth': 'depth'})

group_df_2 = group_df[~group_df['Patient idx'].isin(['90', '177', '38'])]

group_df_2 = group_df_2.rename(columns={'Dice': 'DSC'})
group_df_2 = group_df_2.rename(columns={'filter': 'No. filters'})
group_df_2 = group_df_2.rename(
    columns={'depth': 'Depth'})
plt.subplot(2, 2, 1)
sns.violinplot(x=group_df_2['No. filters'],
               y=group_df_2['DSC'],
               hue=group_df_2['augmentation'],
               cut=0,
               #    order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
               #           for f in [32, 48, 64]],
               hue_order=['No augmentation', 'Affine transform',
                          'Affine transform &\nIntensity Changes'],
               legend=False,
               #    inner='quartile',
               scale='area',
               bw=0.1,
               )
plt.legend().set_visible(False)
# plt.ylim([0.45, 0.9])
plt.subplot(2, 2, 2)
ax = sns.violinplot(x=group_df_2['Depth'],
                    y=group_df_2['DSC'],
                    hue=group_df_2['augmentation'],
                    cut=0,
                    #    order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
                    #           for f in [32, 48, 64]],
                    hue_order=['No augmentation', 'Affine transform',
                               'Affine transform &\nIntensity Changes'],
                    legend=False,
                    #    inner='quartile',
                    # scale='area',
                    # bw=0.1,
                    squeeze=True
                    )
ax.set_ylabel('')
# plt.ylim([0.45, 0.9])
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
           ncol=3, fontsize='small')
plt.show()
group_df_2 = group_df_2.rename(columns={'DSC': 'Dice'})
group_df_2 = group_df_2.rename(columns={'No. filters': 'filter'})
group_df_2 = group_df_2.rename(
    columns={'Depth': 'depth'})


group_df.groupby(['depth', 'augmentation']).agg(
    {'Dice': [Q1, 'median', Q3, 'mean', 'std']})

group_df.groupby(['filter', 'augmentation']).agg(
    {'Dice': [Q1, 'median', Q3, 'mean', 'std']})


sns.set_theme(style='whitegrid')
group_df = group_df.rename(columns={'Dice': 'DSC'})
sns.violinplot(x=group_df['Patient idx'],
               y=group_df['DSC'],
               hue=group_df['augmentation'],
               cut=0,
               #    order=[f'{f} filters\n{d} up/down\nsampling layers' for d in [3, 4]
               #           for f in [32, 48, 64]],
               hue_order=['No augmentation', 'Affine transform',
                          'Affine transform &\nIntensity Changes'],
               legend=False,
               #    inner='quartile',
               scale='area'
               )
plt.legend(loc='lower right')
plt.show()
group_df = group_df.rename(columns={'DSC': 'Dice'})


# Friedman test
# compare samples
no_aug = (group_df['normalize'] != 'FALSE') & (group_df['aug'].isnull())
affine_aug = group_df['aug'] == 'Affine'
default_aug = group_df['aug'] == 'Default'

stat, p = friedmanchisquare(
    group_df[no_aug]['Dice'],
    group_df[affine_aug]['Dice'],
    group_df[default_aug]['Dice'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

df[df.index == df[df['aug'].isnull()]['dice'].idxmax()]
df[df.index == df[df['aug'] == 'Affine']['dice'].idxmax()]
df[df.index == df[df['aug'] == 'Affine']['dice'].idxmax()]
df[df['normalize'] != 'FALSE']['dice'].nlargest()

df[df['normalize'] != 'FALSE']['dice'].idxmax()

all_res = [group_df[group_df.name == name] for name in df.name.values]

stat, p = friedmanchisquare(
    *all_res
)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

group_df[['name', 'filter', 'depth', 'augmentation', 'normalize',
          'Patient idx', 'Dice']].to_csv('csv/Dice_per_patient.csv', index=False)


pval = pd.read_csv('../../pvalue.csv')


pval.head()
pval_columns = pval.columns

sns.heatmap(pval[pval_columns[1:]], xticklabels=pval_columns[1:],
            yticklabels=pval_columns[1:], vmax=0.06)
plt.show()
