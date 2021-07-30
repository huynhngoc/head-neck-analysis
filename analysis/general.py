import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


nrow, ncol = 3, 4
i = 0
result_file = 'csv/2d_results.csv'

df = pd.read_csv(result_file)
# print(df)
# print(df.describe())

mean = df.groupby('filter')['dice'].mean()
std = df.groupby('filter')['dice'].std()

print(mean)
print(std)

df['filter_depth'] = df['filter'].astype(str) + '_' + df['depth'].astype(str)
new_aug_col = df['aug'].copy()
new_aug_col[df['aug'].isnull()] = 'No augmentation'
new_aug_col[df['aug'] == 'Affine'] = 'Affine transform'
df['augmentation'] = new_aug_col

nrow, ncol = 1, 3
i = 0

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['filter'], y=df['dice'])
plt.title('Dice score on different number of filter')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].isnull()]['filter'],
               y=df[df['aug'].isnull()]['dice'])
plt.title('Dice score on different number of filter (no aug)')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].notnull()]['filter'],
               y=df[df['aug'].notnull()]['dice'])
plt.title('Dice score on different number of filter (with aug)')
# plt.show()

plt.show()
# New plot
nrow, ncol = 1, 3
i = 0

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['depth'],
               y=df['dice'])
plt.title('Dice score on different depth')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].isnull()]['depth'],
               y=df[df['aug'].isnull()]['dice'])
plt.title('Dice score on different depth (no aug)')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].notnull()]['depth'],
               y=df[df['aug'].notnull()]['dice'])
plt.title('Dice score on different depth (with aug)')
# plt.show()

plt.show()
# New plot
nrow, ncol = 1, 3
i = 0

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['filter_depth'], y=df['dice'])
plt.title('Dice score on different model complexity')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].isnull()]['filter_depth'],
               y=df[df['aug'].isnull()]['dice'])
plt.title('Dice score on different model complexity (no aug)')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].notnull()]['filter_depth'],
               y=df[df['aug'].notnull()]['dice'])
plt.title('Dice score on different model complexity (with aug)')
# plt.show()

plt.show()
# New plot
nrow, ncol = 2, 3
i = 0


i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['augmentation'], y=df['dice'], )
plt.title('Dice score on different model augmentation')
# plt.show()

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['normalize'], y=df['dice'], )
plt.title('Dice score on different normalization')


i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].isnull()]['normalize'],
               y=df[df['aug'].isnull()]['dice'], )
plt.title('Dice score on different normalization (No Aug)')

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df[df['aug'].notnull()]['normalize'],
               y=df[df['aug'].notnull()]['dice'], )
plt.title('Dice score on different normalization (With Aug)')


plt.show()
# New plot
nrow, ncol = 2, 2
i = 0

i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['filter'], y=df['dice'], hue=df['augmentation'])
plt.title('Dice score on different number of filter')


i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['depth'],
               y=df['dice'],
               hue=df['augmentation'])
plt.title('Dice score on different depth')


i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['filter_depth'],
               y=df['dice'],
               hue=df['augmentation'])
plt.title('Dice score on different model complexity')


i += 1
plt.subplot(nrow, ncol, i)
sns.violinplot(x=df['normalize'],
               y=df['dice'],
               hue=df['augmentation'])
plt.title('Dice score on different normalization')

plt.show()


def Q1(x):
    return x.quantile(0.25)


def Q3(x):
    return x.quantile(0.75)


df.groupby('depth').agg({'dice': [Q1, 'median', Q3, 'mean', 'std']})
df[df.aug.isnull()].groupby('depth').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']})
df[df.aug.notnull()].groupby('depth').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']})


df[df.aug.isnull()].groupby('filter').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']})
df[df.aug.notnull()].groupby('filter').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']})
df.groupby('filter').agg({'dice': [Q1, 'median', Q3, 'mean', 'std']})


df[df.aug.isnull()].groupby('normalize').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']}).sort_index(ascending=False)
df[df.aug.notnull()].groupby('normalize').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']}).sort_index(ascending=False)
df.groupby('normalize').agg(
    {'dice': [Q1, 'median', Q3, 'mean', 'std']}).sort_index(ascending=False)

sns.violinplot(x=df.sort_values('d')['model'],
               y=df.sort_values('d')['dice'],
               hue=df.sort_values('d')['augmentation'],
               legend=False
               )
plt.title('Dice score on different model complexity')
plt.legend(loc='lower right')
plt.show()

df['d'] = df['depth'].astype(str) + '_' + df['filter'].astype(str)
df['model'] = df['filter'].astype(
    str) + ' filters\n' + df['depth'].astype(str) + ' maxpools'
