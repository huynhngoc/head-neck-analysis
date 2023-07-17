import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import h5py
import pandas as pd


def f1_score(y_true, y_pred, beta=1):
    eps = 1e-8

    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    target_positive = np.sum(y_true)
    predicted_positive = np.sum(y_pred)

    fb_numerator = (1 + beta ** 2) * true_positive + eps
    fb_denominator = (
        (beta ** 2) * target_positive + predicted_positive + eps
    )

    return fb_numerator / fb_denominator


def precision(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    # target_positive = np.sum(y_true)
    predicted_positive = np.sum(y_pred)

    return true_positive / predicted_positive


def recall(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    target_positive = np.sum(y_true)
    # predicted_positive = np.sum(y_pred)

    return true_positive / target_positive


def specificity(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_negative = np.sum((1 - y_true) * (1 - y_pred))
    negative_pred = np.sum(1 - y_pred)

    return true_negative / negative_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("source")
    parser.add_argument("--num_mc", default=1, type=int)
    parser.add_argument("--dropout_rate", default=10, type=int)

    args, unknown = parser.parse_known_args()

    base_path = args.source + '/' + args.name + f'_{args.dropout_rate:02d}'
    num_mc = args.num_mc

    print('Base_path:', args.source)
    print('Original model:', args.name)
    print('Dropout rate:', args.dropout_rate)
    print('Number of samples:', num_mc)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    ous_h5 = args.source + '/' + args.name + '/ous_test.h5'
    ous_csv = args.source + '/' + args.name + '/ous_original_results.csv'
    maastro_h5 = args.source + '/' + args.name + '/maastro_full.h5'
    maastro_csv = args.source + '/' + args.name + '/maastro_original_results.csv'
    # model_file = args.source + '/' + args.name + '/model.h5'

    # NOTE: exclude patient 5 from MAASTRO set
    # data = data[data.patient_idx != 5]

    # dropout_model = model_from_full_config(
    #     'config/uncertainty/' + args.name + '_r' + str(args.dropout_rate) + '.json', weights_file=model_file)

    if not os.path.exists(base_path + '/OUS_uncertainty_map'):
        os.makedirs(base_path + '/OUS_uncertainty_map')
    if not os.path.exists(base_path + f'/OUS_uncertainty_map/{num_mc:02d}'):
        os.makedirs(base_path + f'/OUS_uncertainty_map/{num_mc:02d}')

    ous_df = pd.read_csv(ous_csv)

    data = []
    dice_info = []
    iou_info = []
    vol_info = []
    print('Working on OUS.....')
    for pid in ous_df.pid:
        print('PID:', pid)
        # with h5py.File(ous_h5, 'r') as f:
        #     y_true = f['y'][str(pid)][:]
        y_pred = []
        intersection = None
        union = None
        for i in range(1, num_mc + 1):
            print('mc_idx:', i)
            with open(base_path + f'/OUS/{pid}/{i:02d}.npy', 'rb') as f:
                prob = np.load(f)
            # entropy map
            y_pred.append(prob)
            # calculate intersection and union for IoU
            if intersection is None:
                intersection = (prob > 0.5).astype(float)
                union = (prob > 0.5).astype(float)
            else:
                intersection = intersection * (prob > 0.5).astype(float)
                union = union + (prob > 0.5).astype(float)
            # vol
            vol_info.append({
                'pid': pid,
                'mc_idx': i,
                'predicted_vol': (prob > 0.5).sum()
            })

            # cross dice score
            for j in range(i + 1, num_mc + 1):
                if i == j:
                    continue
                print('mc_idx cross:', j)
                with open(base_path + f'/OUS/{pid}/{j:02d}.npy', 'rb') as f:
                    prob_2 = np.load(f)

                dice_info.append({
                    'pid': pid,
                    'mc_idx': f'{i}_{j}',
                    'dice': f1_score((prob > 0.5).astype(float),
                                     (prob_2 > 0.5).astype(float))
                })
                dice_info.append({
                    'pid': pid,
                    'mc_idx': f'{j}_{i}',
                    'dice': f1_score((prob_2 > 0.5).astype(float),
                                     (prob > 0.5).astype(float))
                })

        iou_info.append({
            'pid': pid,
            'iou': intersection.sum() / (union > 0).astype(float).sum()
        })

        y_pred = np.stack(y_pred, axis=0).mean(axis=0)
        uncertainty_map = - y_pred * np.log(y_pred)
        with open(base_path + f'/OUS_uncertainty_map/{num_mc:02d}/{pid}.npy', 'wb') as f:
            np.save(f, uncertainty_map)

    pd.DataFrame(iou_info).to_csv(
        base_path + f'/OUS_analysis/iou_{num_mc:02d}.csv', index=False
    )
    pd.DataFrame(vol_info).to_csv(
        base_path + f'/OUS_analysis/vol_{num_mc:02d}.csv', index=False
    )
    pd.DataFrame(dice_info).to_csv(
        base_path + f'/OUS_analysis/dice_{num_mc:02d}.csv', index=False
    )

    if not os.path.exists(base_path + '/MAASTRO_uncertainty_map'):
        os.makedirs(base_path + '/MAASTRO_uncertainty_map')
    if not os.path.exists(base_path + f'/MAASTRO_uncertainty_map/{num_mc:02d}'):
        os.makedirs(base_path + f'/MAASTRO_uncertainty_map/{num_mc:02d}')

    maastro_df = pd.read_csv(maastro_csv)

    data = []
    dice_info = []
    iou_info = []
    vol_info = []
    print('Working on MAASTRO.....')
    for pid in maastro_df.pid:
        print('PID:', pid)
        # with h5py.File(maastro_h5, 'r') as f:
        #     y_true = f['y'][str(pid)][:]
        y_pred = []
        intersection = None
        union = None
        for i in range(1, num_mc + 1):
            print('mc_idx:', i)
            with open(base_path + f'/MAASTRO/{pid}/{i:02d}.npy', 'rb') as f:
                prob = np.load(f)
            # entropy map
            y_pred.append(prob)
            # calculate intersection and union for IoU
            if intersection is None:
                intersection = (prob > 0.5).astype(float)
                union = (prob > 0.5).astype(float)
            else:
                intersection = intersection * (prob > 0.5).astype(float)
                union = union + (prob > 0.5).astype(float)
            # vol
            vol_info.append({
                'pid': pid,
                'mc_idx': i,
                'predicted_vol': (prob > 0.5).sum()
            })

            # cross dice score
            for j in range(i + 1, num_mc + 1):
                if i == j:
                    continue
                print('mc_idx cross:', j)
                with open(base_path + f'/MAASTRO/{pid}/{j:02d}.npy', 'rb') as f:
                    prob_2 = np.load(f)

                dice_info.append({
                    'pid': pid,
                    'mc_idx': f'{i}_{j}',
                    'dice': f1_score((prob > 0.5).astype(float),
                                     (prob_2 > 0.5).astype(float))
                })
                dice_info.append({
                    'pid': pid,
                    'mc_idx': f'{j}_{i}',
                    'dice': f1_score((prob_2 > 0.5).astype(float),
                                     (prob > 0.5).astype(float))
                })

        iou_info.append({
            'pid': pid,
            'iou': intersection.sum() / (union > 0).astype(float).sum()
        })

        y_pred = np.stack(y_pred, axis=0).mean(axis=0)
        uncertainty_map = - y_pred * np.log(y_pred)
        with open(base_path + f'/MAASTRO_uncertainty_map/{num_mc:02d}/{pid}.npy', 'wb') as f:
            np.save(f, uncertainty_map)

    pd.DataFrame(iou_info).to_csv(
        base_path + f'/MAASTRO_analysis/iou_{num_mc:02d}.csv', index=False
    )
    pd.DataFrame(vol_info).to_csv(
        base_path + f'/MAASTRO_analysis/vol_{num_mc:02d}.csv', index=False
    )
    pd.DataFrame(dice_info).to_csv(
        base_path + f'/MAASTRO_analysis/dice_{num_mc:02d}.csv', index=False
    )
