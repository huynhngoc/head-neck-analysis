from deoxys.utils import read_csv
from sklearn.metrics import roc_auc_score
import h5py
import pandas as pd
import numpy as np
import argparse


def from_postfix(postfix):
    if postfix:
        return int(postfix[-1])
    else:
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args, unknown = parser.parse_known_args()

    name_template = '3d_{input}_f{fold}_{output}_{optimizer}'
    log_path = '/net/fs-1/Ngoc/hnperf/{name}{postfix}/logs.csv'
    prediction_path = '/mnt/SCRATCH/ngochuyn/hnperf/{name}{postfix}/prediction/prediction.{epoch:03d}.h5'

    results = []

    for fold in range(5):
        for postfix in ['', '_2', '_3', '_4', '_5']:
            for optimizer in ['SGD', 'adam']:
                name = name_template.format(
                    input=args.input, fold=fold,
                    output=args.output, optimizer=optimizer)
                df = read_csv(log_path.format(name=name, postfix=postfix))
                best_auc = df['val_auc'].max()
                best_epoch = df['val_auc'].argmax() + 1

                with h5py.File(
                        prediction_path.format(
                        name=name, postfix=postfix,
                        epoch=best_epoch), 'r') as f:
                    y = f['y'][:]
                    predicted = f['predicted'][:]
                real_auc = roc_auc_score(y, predicted)

                results.append(
                    [name, args.input, args.output, optimizer, fold,
                     from_postfix(postfix), best_epoch, best_auc, real_auc
                     ])
    pd.DataFrame(np.array(results), columns=[
        'name', 'input', 'output', 'optimizer', 'fold', 'runs', 'best_epoch',
        'est_auc', 'auc'
    ]).to_csv(f'outcome_res/{args.input}_{args.output}.csv')
