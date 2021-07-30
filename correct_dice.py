import argparse
import pandas as pd
import numpy as np
import h5py
import os
import shutil
import h5py
from deoxys.experiment import postprocessor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model")

    args, unknown = parser.parse_known_args()

    test_index = [5, 8, 13, 16, 18, 21, 36, 44, 52, 55,
                  60, 61, 67, 73, 74, 77, 82,
                  91, 93, 99, 110, 116, 120, 130, 140,
                  148, 153, 154, 162, 164, 169, 184, 191,
                  194, 209, 217, 223, 233, 242, 249]

    val_index = [29, 35, 38, 49, 70, 87, 90, 98,
                 163, 170, 177, 213, 229, 241, 246]

    base = '/net/fs-1/Ngoc/hnperf_2/'

    model = args.model

    # val_prob = '/net/fs-1/Ngoc/hnperf/binary_2d_CT_W_PET_aug_new_sif/prediction/prediction.028.h5'
    test_prob = '/net/fs-1/Ngoc/hnperf/binary_2d_CT_W_PET_aug_new_sif/test/prediction_test.h5'
    val_df = pd.DataFrame(val_index, columns=['patient_idx'])
    test_df = pd.DataFrame(test_index, columns=['patient_idx'])

    val_df.to_csv(base + model + '/corrected/val_res_2.csv', index=False)
    test_df.to_csv(base + model + '/corrected/test_res_2.csv', index=False)

    print(model)
    raw_file = base + model + '/test/prediction_test.h5'
    corrected_file = base + model + '/corrected/prediction_test_2.h5'
    shutil.copy(raw_file, corrected_file)
    #print('Putting corrected data in val group')
    #for pid in val_index:
    #    print('patient', pid)
    #    with h5py.File(val_prob, 'r') as f:
    #        prob = f['predicted'][str(pid)][:]
    #    prob = np.array([np.full((191, 265, 1), p[0]) for p in prob])
    #    with h5py.File(corrected_file, 'a') as f:
    #        pred = f['predicted'][str(pid)][:]
    #        f['predicted'][str(pid)][:] = pred * prob
    print('Putting corrected data in test + val group')
    for pid in val_index + test_index:
        print('patient', pid)
        with h5py.File(test_prob, 'r') as f:
            prob = f['predicted'][str(pid)][:]
        prob = np.array([np.full((191, 265, 1), p[0]) for p in prob])
        with h5py.File(corrected_file, 'a') as f:
            pred = f['predicted'][str(pid)][:]
            f['predicted'][str(pid)][:] = pred * prob
    print('recalculate dice score')

    postprocessor.H5CalculateFScore(
        corrected_file,
        base + model + '/corrected/test_res_2.csv',
        map_file=base + model + '/corrected/test_res_2.csv',
        map_column='patient_idx'
    ).post_process()
    postprocessor.H5CalculateFScore(
        corrected_file,
        base + model + '/corrected/val_res_2.csv',
        map_file=base + model + '/corrected/val_res_2.csv',
        map_column='patient_idx'
    ).post_process()
