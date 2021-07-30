import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

result_file = 'csv/2d_results.csv'
path_file = 'csv/paths.csv'

dice_per_slice_path = '/single_map/logs.{epoch:03d}.csv'
dice_per_patient_path = '/logs/logs.{epoch:03d}.csv'
analysis_path = '/analysis'


markers = ['o-', 'v-', '^-', '<-', '>-',
           '1-', '2-', 's-', 'p-', 'P-',
           '*-', '+-', 'x-', 'D-', 'd-', '--']


def epoch_from_filename(filename):
    extension_index = filename.rindex('.')

    return int(filename[extension_index-3:extension_index])


path_df = pd.read_csv(path_file)
path = path_df[path_df['type'] == '2d']['path'].values[0]
print(path)

experiments = pd.read_csv(result_file)
# experiments['filter_depth'] = experiments['filter'].astype(
#     str) + '_' + experiments['depth'].astype(str)
# new_aug_col = experiments['aug'].copy()
# new_aug_col[experiments['aug'].isnull()] = 'No augmentation'
# new_aug_col[experiments['aug'] == 'Affine'] = 'Affine transform'
# experiments['augmentation'] = new_aug_col

dice_per_exp = []
dice_per_exp_2 = []
recal_dice = []

for exp, dice in zip(experiments['name'], experiments['dice']):
    save_path = path + exp + analysis_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    best_epoch = epoch_from_filename(os.listdir(path + exp + '/prediction')[0])

    # ##########################################################################
    # Plot dice per dice each patient
    dice_per_slice = pd.read_csv(
        path + exp + dice_per_slice_path.format(epoch=best_epoch))
    plt.figure(figsize=(10, 8))
    sns.violinplot(x=dice_per_slice['patient_idx'],
                   y=dice_per_slice['f1_score'], cut=0)
    plt.title(
        f'Model {exp} - Epoch {best_epoch} - Mean Dice {dice:.6f}'
        '\nDice score per slice of different patient')
    plt.savefig(save_path + '/dice_per_slice.png')
    plt.close('all')
    # plt.show()
    # End plotting
    # ##########################################################################

    # ##########################################################################
    # Get dice per epoch
    patient_dice_per_epoch = []
    epoch_results = os.listdir(path + exp + '/logs')
    epochs = []
    for res_file in epoch_results:
        data = pd.read_csv(path + exp + '/logs/' + res_file)
        epoch = epoch_from_filename(res_file)
        epochs.append(epoch)

        patient_dice_per_epoch.append(
            data['f1_score'].values)

    # end get dice per epoch
    # ##########################################################################

    best_patient_dice_df = pd.read_csv(
        path + exp + dice_per_patient_path.format(epoch=best_epoch))

    # ##########################################################################
    # Plot dice per epoch
    patient_idx = best_patient_dice_df['patient_idx'].values

    # print(patient_dice_per_epoch)
    all_data = np.vstack(patient_dice_per_epoch)

    df = pd.DataFrame(all_data, columns=patient_idx)
    df.index = epochs
    df.index.name = 'epoch'
    # df['mean'] = df.mean(axis=1)
    df['mean'] = df[[pid for pid in patient_idx if pid != 90]].mean(axis=1)
    best_epoch_2 = df['mean'].idxmax()

    plt.figure(figsize=(10, 8))
    df.plot(style=markers[:16], ax=plt.gca())
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title(
        f'Model {exp} \nBest Epoch {best_epoch_2} - Mean Dice {dice:.6f}')
    plt.savefig(save_path + '/dice_per_epoch_2.png')
    plt.close('all')

    df.to_csv(save_path + '/log_recal.csv')

    best_patient_dice_df_2 = pd.read_csv(
        path + exp + dice_per_patient_path.format(epoch=best_epoch_2))
    dice_per_exp_2.append(best_patient_dice_df_2['f1_score'].values)
    recal_dice.append(df['mean'].max())
    # plt.show()
    # End plot
    # ##########################################################################

    dice_per_exp.append(best_patient_dice_df['f1_score'].values)

all_data = np.vstack(dice_per_exp)
print(all_data.shape)
for i, pid in enumerate(best_patient_dice_df['patient_idx'].values):
    experiments[str(pid)] = all_data[:, i]

print(experiments)
experiments.to_csv('csv/all_res.csv', index=False)

experiments = pd.read_csv(result_file)
all_data = np.vstack(dice_per_exp_2)
experiments['DSC'] = recal_dice
for i, pid in enumerate(best_patient_dice_df_2['patient_idx'].values):
    experiments[str(pid)] = all_data[:, i]
experiments.to_csv('csv/all_res_recal.csv', index=False)
