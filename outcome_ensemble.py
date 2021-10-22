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
    parser.add_argument("output")

    args, unknown = parser.parse_known_args()

    name_template = '3d_{input}_f{fold}_{output}_{optimizer}'
    log_path = '/net/fs-1/Ngoc/hnperf/{name}{postfix}/logs.csv'
    prediction_path = '/mnt/SCRATCH/ngochuyn/hnperf/{name}{postfix}/prediction/prediction.{epoch:03d}.h5'

    results = []

    for fold in range(5):
        fold_df = None
        for inp in ['clinical', 'shape', 'texture', 'radiomic']:
            best_real_auc = 0
            best_predicted = None
            for optimizer in ['SGD', 'adam']:
                for postfix in ['', '_2', '_3', '_4', '_5']:
                    name = name_template.format(
                        input=inp, fold=fold,
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
                    if real_auc > best_real_auc:
                        best_real_auc = real_auc
                        best_predicted = predicted
                        info = [name, inp, args.output,
                                optimizer, fold,
                                from_postfix(postfix), best_epoch,
                                best_auc, real_auc]

            results.append(info)
            if fold_df is None:
                fold_df = pd.DataFrame(np.array([y]), columns=['y'])
            fold_df[inp] = best_predicted
        df.to_csv(f'outcome_res/f{fold}_{args.output}.csv', index=False)
    pd.DataFrame(np.array(results), columns=[
        'name', 'input', 'output', 'optimizer', 'fold', 'runs', 'best_epoch',
        'est_auc', 'auc'
    ]).to_csv(f'outcome_res/all_{args.output}.csv', index=False)
