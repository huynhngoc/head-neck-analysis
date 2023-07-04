import matplotlib.pyplot as plt
from deoxys.customize import custom_layer
from deoxys.model import load_model
from deoxys.customize import custom_layer
from deoxys.model.model import model_from_full_config
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.keras.backend import dropout
import tensorflow as tf
import argparse
import os
import h5py
import pandas as pd


@custom_layer
class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


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
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("source")
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--dropout_rate", default=10, type=int)

    args, unknown = parser.parse_known_args()

    base_path = args.source + '/' + args.name + '_' + str(args.dropout_rate)
    iter = args.iter

    print('Base_path:', args.source)
    print('Original model:', args.name)
    print('Dropout rate:', args.dropout_rate)
    print('Iteration:', iter)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    ous_h5 = args.source + '/' + args.name + '/ous_test.h5'
    ous_csv = args.source + '/' + args.name + '/ous_test.csv'
    maastro_h5 = args.source + '/' + args.name + '/maastro_full.h5'
    maastro_csv = args.source + '/' + args.name + '/maastro_full.csv'
    # model_file = args.source + '/' + args.name + '/model.h5'

    # NOTE: exclude patient 5 from MAASTRO set
    # data = data[data.patient_idx != 5]

    # dropout_model = model_from_full_config(
    #     'config/uncertainty/' + args.name + '_r' + str(args.dropout_rate) + '.json', weights_file=model_file)

    if not os.path.exists(base_path + '/OUS_analysis'):
        os.makedirs(base_path + '/OUS_analysis')

    ous_df = pd.read_csv(ous_csv)

    data = []
    print('Working on OUS.....')
    for pid in ous_df.patient_idx:
        print('PID:', pid)
        with h5py.File(maastro_h5, 'r') as f:
            y_true = f['y'][str(pid)][:]

        y_pred = []
        for i in range(1, iter+1):
            with open(base_path + '/OUS/' + str(pid) + f'/{iter:02d}.npy', 'rb') as f:
                y_pred.append(np.load(f)[0])
        y_pred = np.stack(y_pred, axis=0).mean(axis=0)

        data.append({
            'pid': pid,
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'specificity': specificity(y_true, y_pred)
        })

    pd.DataFrame(data).to_csv(
        base_path + f'/OUS_analysis/average_{iter:02d}.csv', index=False
    )

    if not os.path.exists(base_path + '/MAASTRO_analysis'):
        os.makedirs(base_path + '/MAASTRO_analysis')

    data = []
    maastro_df = pd.read_csv(maastro_csv)
    print('Working on MAASTRO.....')
    for pid in maastro_df.patient_idx:
        print('PID:', pid)
        with h5py.File(maastro_h5, 'r') as f:
            y_true = f['x'][str(pid)][:]

        y_pred = []
        for i in range(1, iter+1):
            with open(base_path + '/MAASTRO/' + str(pid) + f'/{iter:02d}.npy', 'rb') as f:
                y_pred.append(np.load(f)[0])
        y_pred = np.stack(y_pred, axis=0).mean(axis=0)

        data.append({
            'pid': pid,
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'specificity': specificity(y_true, y_pred)
        })

    pd.DataFrame(data).to_csv(
        base_path + f'/OUS_analysis/average_{iter:02d}.csv', index=False
    )
