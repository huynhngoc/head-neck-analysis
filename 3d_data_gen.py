import h5py
import numpy as np
import gc


if __name__ == '__main__':
    # test_index = [5, 8, 13, 16, 18, 21, 36, 44, 52, 55,
    #               60, 61, 67, 73, 74, 77, 82,
    #               91, 93, 99, 110, 116, 120, 130, 140,
    #               148, 153, 154, 162, 164, 169, 184, 191,
    #               194, 209, 217, 223, 233, 242, 249]

    # val_index = [29, 35, 38, 49, 70, 87, 90, 98,
    #              163, 170, 177, 213, 229, 241, 246]

    # # all_index = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15, 16,
    # #              18, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31,
    # #              32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44,
    # #              45, 48, 49, 50, 52, 54, 55, 56, 57, 60, 61,
    # #              62, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74,
    # #              77, 78, 81, 82, 83, 86, 87, 88, 89, 90, 91,
    # #              92, 93, 94, 95, 96, 97, 98, 99, 100, 102,
    # #              103, 104, 105, 106, 107, 108, 109, 110, 111,
    # #              112, 113, 114, 115, 116, 117, 118, 120, 121,
    # #              123, 124, 125, 126, 127, 128, 129, 130, 131,
    # #              133, 136, 138, 139, 140, 141, 142, 143, 144,
    # #              146, 147, 148, 149, 150, 151, 152, 153, 154,
    # #              155, 156, 157, 158, 159, 161, 162, 163, 164,
    # #              165, 166, 167, 168, 169, 170, 171, 172, 173,
    # #              176, 177, 178, 180, 181, 182, 184, 185, 187,
    # #              189, 191, 194, 195, 196, 197, 198, 199, 200,
    # #              201, 202, 203, 204, 205, 206, 207, 209, 210,
    # #              212, 213, 215, 216, 217, 218, 220, 222, 223,
    # #              224, 225, 228, 229, 230, 231, 232, 233, 239,
    # #              240, 241, 242, 243, 244, 246, 247, 248, 249,
    # #              250, 252, 253, 254]

    # # train_index = [item for item in all_index
    # #                if item not in val_index and item not in test_index]

    # train_index = [23, 247, 131, 207, 181, 176, 156, 173, 37, 107,
    #                239, 158, 62, 112, 81, 243, 199, 108, 14, 231, 240,
    #                150, 124, 157, 105, 189, 200, 196, 147, 149, 171,
    #                254, 250, 165, 30, 178, 224, 230, 248, 121, 168,
    #                146, 216, 109, 139, 50, 210, 228, 111, 24, 129,
    #                57, 32, 127, 39, 225, 96, 138, 133, 10, 232, 64,
    #                159, 15, 106, 88, 123, 128, 252, 25, 2, 94, 115,
    #                166, 83, 167, 187, 222, 215, 66, 22, 136, 114,
    #                126, 78, 95, 205, 172, 117, 89, 48, 31, 244, 185,
    #                253, 125, 54, 103, 155, 72, 65, 86, 201, 27, 143,
    #                118, 4, 42, 151, 104, 68, 141, 182, 197, 12, 71,
    #                142, 11, 102, 26, 56, 152, 180, 161, 212, 100,
    #                113, 144, 218, 34, 97, 198, 43, 45, 92, 202, 220,
    #                40, 195, 203, 204, 206]

    # fold_idx = 0

    # for i in range(0, len(train_index), 15):
    #     with h5py.File('../../headneck_3d_new.h5', 'a') as hf:
    #         group = hf.create_group('fold_{}'.format(fold_idx))
    #         group.create_dataset('patient_idx', data=train_index[i:i+15])

    #     fold_idx += 1

    # for i in range(0, len(val_index), 15):
    #     with h5py.File('../../headneck_3d_new.h5', 'a') as hf:
    #         group = hf.create_group('fold_{}'.format(fold_idx))
    #         group.create_dataset('patient_idx', data=val_index[i:i+15])

    #     fold_idx += 1

    # for i in range(0, len(test_index), 15):
    #     with h5py.File('../../headneck_3d_new.h5', 'a') as hf:
    #         group = hf.create_group('fold_{}'.format(fold_idx))
    #         group.create_dataset('patient_idx', data=test_index[i:i+15])

    #     fold_idx += 1

    path_to_patients = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/Dataset_3D_U_NET/imdata/imdata_P{:03d}.mat'

    for fold_idx in range(15):
        gc.collect()
        print('Working on fold', fold_idx)
        with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
            patient_idx = hf[f'fold_{fold_idx}']['patient_idx'][:]
        inputs = np.zeros((len(patient_idx), 173, 191, 265, 2))
        targets = np.zeros((len(patient_idx), 173, 191, 265, 1))

        print(inputs.shape)

        for i, p_idx in enumerate(patient_idx):
            gc.collect()
            with h5py.File(
                    path_to_patients.format(p_idx), 'r') as patient_file:
                imdata = patient_file['imdata']
                CT = imdata['CT'][:]
                PET = imdata['PT'][:]
                final_image = np.stack([CT, PET], axis=-1).squeeze()
                tumor = imdata['tumour'][:]
                nodes = imdata['nodes'][:]
                mask = np.logical_or(tumor, nodes).squeeze().astype(
                    'float32')[..., np.newaxis]

                print(final_image.shape, mask.shape)

                if final_image.shape == inputs.shape[1:]:
                    inputs[i] = final_image
                    targets[i] = mask
                else:
                    print('Shape mismatch, re-arranging...')
                    x, y, z = final_image.shape[:-1]
                    start_x = (173 - x) // 2
                    start_y = (191 - y) // 2
                    start_z = (265 - z) // 2
                    inputs[i, start_x: start_x+x, start_y: start_y +
                           y, start_z:start_z+z, :] = final_image
                    targets[i, start_x: start_x+x, start_y: start_y +
                            y, start_z:start_z+z, :] = mask

        with h5py.File('../../headneck_3d_new.h5', 'a') as hf:
            group = hf[f'fold_{fold_idx}']
            group.create_dataset('input', data=inputs,
                                 dtype='f4', chunks=(1, 173, 191, 265, 2),
                                 compression='lzf')
            group.create_dataset('target', data=targets,
                                 dtype='f4', chunks=(1, 173, 191, 265, 1),
                                 compression='lzf')
