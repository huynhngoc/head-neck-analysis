import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
path = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/HN_MAASTRICT_29062020_Matlab/imdata_cropped'
main_folder = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/HN_MAASTRICT_29062020_Matlab'
map_file = '/patient_names_mapping.csv'
newpath = '//nmbu.no/largefile/Project/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/HN_MAASTRICT_29062020_Matlab/imdata_cropped_new'

# patient_files = os.listdir(path)

# # Check size
# for patient_file in patient_files:
#     print(patient_file)
#     with h5py.File(path + '/' + patient_file, 'r') as f:
#         # print(f.keys())
#         # print(f['imdata'].keys())
#         # CT = f['imdata']['CT'][:]
#         assert f['imdata']['CT'].shape == (173, 191, 265)
#         assert f['imdata']['PT'].shape == (173, 191, 265)
#         assert f['imdata']['nodes'].shape == (173, 191, 265)
#         assert f['imdata']['tumour'].shape == (173, 191, 265)


# Create mapping file
# name_template = 'PMA{:03d}'
# total_p = len(patient_files)

# names = [patient_file[:-4].split('_')[-1] for patient_file in patient_files]

# mapping_names = [name_template.format(i) for i in range(1, total_p + 1)]
# mapping_names
# pd.DataFrame(
#     {'name': mapping_names, 'origin':names, 'file': patient_files}
#     ).to_csv(main_folder + '/patient_names_mapping.csv', index=False)


df = pd.read_csv(main_folder + map_file)

for i in range(8):
    print('working on fold', i)
    start, end = i*15, (i+1)*15
    patient_idx = df['name'].values[start: end]
    total_fold = len(patient_idx)

    inputs = np.zeros((total_fold, 173, 191, 265, 2))
    targets = np.zeros((total_fold, 173, 191, 265, 1))

    print('reading files')
    for j, filename in enumerate(df['file'].values[start: end]):
        with h5py.File(path + '/' + filename, 'r') as f:
            imdata = f['imdata']
            CT = imdata['CT'][:]
            PET = imdata['PT'][:]
            final_image = np.stack([CT, PET], axis=-1).squeeze()
            tumor = imdata['tumour'][:]
            nodes = imdata['nodes'][:]
            mask = np.logical_or(tumor, nodes).squeeze().astype(
                'float32')[..., np.newaxis]
        inputs[j] = final_image
        targets[j] = mask

    print('creating the fold')
    with h5py.File('../../maastro_test_data_3d.h5', 'a') as hf:
        group = hf.create_group(f'fold_{i}')
        group.create_dataset('input', data=inputs,
                             dtype='f4', chunks=(1, 173, 191, 265, 2),
                             compression='lzf')
        group.create_dataset('target', data=targets,
                             dtype='f4', chunks=(1, 173, 191, 265, 1),
                             compression='lzf')
        group.create_dataset('patient_idx', data=patient_idx)


for index, (name, origin, filename, mapping_name) in df.iterrows():
    os.rename(newpath + '/' + mapping_name + '.mat', newpath + '/' + filename)


for i in range(8):
    print('working on fold', i)
    start, end = i*15, (i+1)*15
    patient_idx = df['name'].values[start: end]
    total_fold = len(patient_idx)

    inputs = np.zeros((total_fold*173, 191, 265, 2))
    targets = np.zeros((total_fold*173, 191, 265, 1))

    print('reading files')
    for j, filename in enumerate(df['file'].values[start: end]):
        with h5py.File(path + '/' + filename, 'r') as f:
            imdata = f['imdata']
            CT = imdata['CT'][:]
            PET = imdata['PT'][:]
            final_image = np.stack([CT, PET], axis=-1).squeeze()
            tumor = imdata['tumour'][:]
            nodes = imdata['nodes'][:]
            mask = np.logical_or(tumor, nodes).squeeze().astype(
                'float32')[..., np.newaxis]
        inputs[j*173:(j+1) * 173] = final_image
        targets[j*173:(j+1) * 173] = mask

    print('creating the fold')
    with h5py.File('../../maastro_test_data_2d.h5', 'a') as hf:
        group = hf.create_group(f'fold_{i}')
        group.create_dataset('input', data=inputs,
                             dtype='f4', chunks=(1, 191, 265, 2),
                             compression='lzf')
        group.create_dataset('target', data=targets,
                             dtype='f4', chunks=(1, 191, 265, 1),
                             compression='lzf')
        group.create_dataset('patient_idx',
                             data=np.repeat(patient_idx, 173))
