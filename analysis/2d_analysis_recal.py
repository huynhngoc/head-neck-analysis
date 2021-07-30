import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

result_file = 'csv/all_res_recal.csv'

df = pd.read_csv(result_file)
df.columns = [str(col) for col in df.columns]

filter_depth = df['filter'].astype(str) + '_' + df['depth'].astype(str)
df.insert(loc=len(df.columns) - 15, column='filter_depth', value=filter_depth)

new_aug_col = df['aug'].copy()
new_aug_col[df['aug'].isnull()] = 'No augmentation'
new_aug_col[df['aug'] == 'Affine'] = 'Affine transform'
df.insert(loc=len(df.columns) - 15, column='augmentation', value=new_aug_col)

sns.violinplot(x=df['filter_depth'], y=df['DSC'], hue=df['augmentation'],
               order=[
               '32_3', '48_3', '64_3', '32_4', '48_4', '64_4'],
               inner='quartiles')
plt.title('Dice score on different model complexity')
plt.show()

group_df = df.melt(id_vars=df.columns[:-15],
                   var_name='Patient idx', value_name='Dice')


def Q1(x):
    return x.quantile(0.25)


def Q3(x):
    return x.quantile(0.75)


def to_int(x):
    return x.astype(int)


best_models = df['DSC'].nlargest(5).index
df[df.index.isin(best_models)]

group_df.groupby('Patient idx').agg(
    {'Dice': ['min', Q1, 'median', Q3, 'max', 'mean', 'std']}).sort_index(key=to_int)


sns.violinplot(x=group_df['Patient idx'],
               y=group_df['Dice'],
               hue=group_df['augmentation'],
               cut=0
               )
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
