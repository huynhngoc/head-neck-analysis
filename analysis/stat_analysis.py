import pingouin as pg
import pandas as pd

# df = pd.read_csv('csv/dice_per_patient.csv')
df = pd.read_csv('csv/corrected_val_res.csv')

# add norm
norms = []
for name in df.name:
    if 'norm' in name:
        norms.append('norm')
    elif 'P10' in name:
        norms.append('P10')
    elif 'P25' in name:
        norms.append('P25')
    else:
        norms.append('None')
has_norm = [norm != 'None' for norm in norms]
df['norm'] = norms
df['is_norm'] = has_norm
# add aug
augs = []
for name in df.name:
    if 'affine' in name:
        augs.append('affine')
    elif 'aug' in name:
        augs.append('aug')
    else:
        augs.append('None')
has_aug = [norm != 'None' for norm in augs]
df['aug'] = augs
df['is_aug'] = has_aug
# add filter
filters = []
for name in df.name:
    if '32' in name:
        filters.append(32)
    elif '48' in name:
        filters.append(48)
    else:
        filters.append(64)
df['filters'] = filters
# add depth
depth = []
for name in df.name:
    if 'D3' in name:
        depth.append(3)
    else:
        depth.append(4)
df['depth'] = depth

pg.friedman(data=df, dv='f1_score', within='aug',
            subject='patient_idx')
base = ['2d_unet', '2d_unet_32', '2d_unet_32_D3',
        '2d_unet_48', '2d_unet_48_D3', '2d_unet_64_D3']


compare = [name+'_norm' for name in base]

pg.friedman(data=df[df.name.isin(compare)], dv='f1_score', within='filters',
            subject='patient_idx')
pg.friedman(data=df[df.name.isin(compare)], dv='f1_score', within='depth',
            subject='patient_idx')
pg.pairwise_ttests(df[df.name.isin(compare)], dv='f1_score',
                   within=['filters', 'depth'],
                   subject='patient_idx', parametric=False,
                   effsize='r',  padjust='bonf',
                   # between='augmentation'
                   )

pg.pairwise_ttests(df, dv='f1_score',
                   within='filters',
                   subject='patient_idx', parametric=False,
                   effsize='r',  padjust='bonf',
                   # between='augmentation'
                   ).round(2)


res.sort_values('p-unc')
res[res['p-unc'] < 0.05]


x1 = df[df.name == '2d_unet_P25']
x2 = df[df.name == '2d_unet_P25_aug']
post_hoc = df[df.name.isin(
    ['2d_unet_P25', '2d_unet_P25_aug', '2d_unet_P25_aug_affine'])]
pg.wilcoxon(x1.f1_score, x2.f1_score)
pg.pairwise_ttests(df[~df.augmentation.isin(['No augmentation'])], dv='Dice',
                   within='normalize',
                   subject='Patient idx', parametric=False,
                   effsize='cohen', padjust='bonf',
                   # between='augmentation'
                   )

res = pg.pairwise_ttests(df[df.name.isin(
    ['2d_unet', '2d_unet_32', '2d_unet_32_D3', '3f'])], dv='Dice',
    within='name',
    subject='Patient idx', parametric=False,
    effsize='cohen', padjust='holm',
    #    between='augmentation'
)
res
res[res['p-unc'] < 0.01].sort_values('cohen', ascending=False)
res[res['p-corr'] < 0.05]
selected = (df.filter == 32) & (df.depth == 4)
filter_32 = df['filter'] == 32
depth_4 = df['depth'] == 4
df[filter_32 & depth_4]
res = pg.pairwise_ttests(df[filter_32 & depth_4], dv='Dice',
                         within='name',
                         subject='Patient idx', parametric=False,
                         effsize='cohen', padjust='holm',
                         #    between='augmentation'
                         )
res[res['p-unc'] < 0.05]


df = pd.read_csv('csv/corrected_val_results.csv')
res = df.groupby('name').agg({'f1_score': ['mean']})

df_score = pd.DataFrame(res.index, columns=['name'])
df_score['dice'] = res.values.flatten()
df_score.sort_values('dice', ascending=False)


df = pd.read_csv('csv/corrected_test_results.csv')
res = df.groupby('name').agg({'f1_score': ['mean']})

df_score = pd.DataFrame(res.index, columns=['name'])
df_score['dice'] = res.values.flatten()
df_score.sort_values('dice', ascending=False)
