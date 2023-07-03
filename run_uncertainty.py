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


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("save_path")
    parser.add_argument("source")
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--dropout_rate", default=10, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    base_path = args.save_path + '/' + args.name + '_' + str(args.dropout_rate)
    iter = args.iter

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    ous_h5 = args.source + '/ous_test.h5'
    ous_csv = args.source + '/ous_test.csv'
    maastro_h5 = args.source + '/maastro_full.h5'
    maastro_csv = args.source + '/maastro_full.csv'
    model_file = args.source + '/model.h5'

    # NOTE: exclude patient 5 from MAASTRO set
    # data = data[data.patient_idx != 5]

    dropout_model = model_from_full_config(
        'config/uncertainty/' + args.name + '_r' + str(args.dropout_rate) + '.json', weights_file=model_file)

    if not os.path.exists(base_path + '/OUS'):
        os.makedirs(base_path + '/OUS')

    ous_df = pd.read_csv(ous_csv)

    for pid in ous_df.patient_idx:
        if not os.path.exists(base_path + '/' + str(pid)):
            os.makedirs(base_path + '/OUS/' + str(pid))
        with h5py.File(ous_h5, 'r') as f:
            images = f['x'][str(pid)][:][None, ...]
        preds = dropout_model.predict(images)

        with open(base_path + '/' + str(pid) + f'/{iter:02d}.npy') as f:
            np.save(f, preds[0])

    if not os.path.exists(base_path + '/MAASTRO'):
        os.makedirs(base_path + '/MAASTRO')

    maastro_df = pd.read_csv(maastro_csv)

    for pid in maastro_df.patient_idx:
        if not os.path.exists(base_path + '/MAASTRO/' + str(pid)):
            os.makedirs(base_path + '/' + str(pid))
        with h5py.File(ous_h5, 'r') as f:
            images = f['x'][str(pid)][None, ...]
        preds = dropout_model.predict(images)

        with open(base_path + '/' + str(pid) + f'/{iter:02d}.npy') as f:
            np.save(f, preds[0])
