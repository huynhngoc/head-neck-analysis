import h5py


with h5py.File('../../maastro_test_data_2d_cut.h5', 'w') as hf:
    for i in range(8):
        hf.create_group(f'fold_{i}')

for i in range(8):
    with h5py.File('../../maastro_test_data_2d.h5', 'r') as hf:
        target = hf[f'fold_{i}']['target'][:]
        inputs = hf[f'fold_{i}']['input'][:]
        patient_idx = hf[f'fold_{i}']['patient_idx'][:]
        target_index = target.sum(axis=(1, 2, 3)) > 0
    with h5py.File('../../maastro_test_data_2d_cut.h5', 'a') as f:
        group = f[f'fold_{i}']
        group.create_dataset('input', data=inputs[target_index],
                             dtype='f4', chunks=(1, 191, 265, 2),
                             compression='lzf')
        group.create_dataset('target', data=target[target_index],
                             dtype='f4', chunks=(1, 191, 265, 1),
                             compression='lzf')
        group.create_dataset('patient_idx',
                             data=patient_idx[target_index])


# with h5py.File('../../maastro_test_data_2d_cut.h5', 'a') as hf:
#     group = hf.create_group(f'fold_{i}')
#     group.create_dataset('input', data=inputs,
#                          dtype='f4', chunks=(1, 191, 265, 2),
#                          compression='lzf')
#     group.create_dataset('target', data=targets,
#                          dtype='f4', chunks=(1, 191, 265, 1),
#                          compression='lzf')
#     group.create_dataset('patient_idx',
#                          data=np.repeat(patient_idx, 173))
