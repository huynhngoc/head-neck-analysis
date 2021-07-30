from skimage import measure
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np

file_3d = '../../headneck_3d_new.h5'

# with h5py.File('../../headneck_2d_full.h5', 'w') as f:
#     for i in range(14):
#         f.create_group(f'fold_{i}')


# def get_slice_idx(patients):
#     idx = [0]
#     for i, pid in enumerate(patients[1:]):
#         if pid == patients[i]:
#             idx.append(idx[-1] + 1)
#         else:
#             idx.append(0)
#     return idx


# for i in range(14):
#     print(f'fold_{i}')
#     with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#         x = np.vstack(hf[f'fold_{i}']['input'][:])
#         y = np.vstack(hf[f'fold_{i}']['target'][:])
#         pid = np.repeat(hf[f'fold_{i}']['patient_idx'][:], 173)

#     with h5py.File('../../headneck_2d_full.h5', 'a') as f:
#         f[f'fold_{i}'].create_dataset('input', data=x,
#                                       dtype='f4', chunks=(1, 191, 265, 1),
#                                       compression='lzf')
#         f[f'fold_{i}'].create_dataset('target', data=y,
#                                       dtype='f4', chunks=(1, 191, 265, 1),
#                                       compression='lzf')
#         f[f'fold_{i}'].create_dataset('patient_idx', data=pid)

#         f[f'fold_{i}'].create_dataset('slice_idx', data=get_slice_idx(pid))

for i in range(14):
    with h5py.File('../../headneck_2d_full.h5', 'a') as f:
        y = f[f'fold_{i}']['target'][:]
        single_class = (y.sum(axis=(1, 2, 3)) > 0).astype(int)
        f[f'fold_{i}'].create_dataset('class', data=single_class)

# # this will replace the file
# with h5py.File('../../headneck_2d_new.h5', 'w') as f:
#     f.create_group('fold_10')
#     f.create_group('fold_11')
#     f.create_group('fold_12')
#     f.create_group('fold_13')


# with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#     x = np.vstack(hf['fold_11']['input'][:])
#     y = np.vstack(hf['fold_11']['target'][:])
#     pid = np.repeat(hf['fold_11']['patient_idx'][:], 173)

# with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#     f['fold_11'].create_dataset('input', data=x,
#                                 dtype='f4', chunks=(1, 191, 265, 2),
#                                 compression='lzf')
#     f['fold_11'].create_dataset('target', data=y,
#                                 dtype='f4', chunks=(1, 191, 265, 1),
#                                 compression='lzf')
#     f['fold_11'].create_dataset('patient_idx', data=pid)


# with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#     x = np.vstack(hf['fold_12']['input'][:])
#     y = np.vstack(hf['fold_12']['target'][:])
#     pid = np.repeat(hf['fold_12']['patient_idx'][:], 173)

# with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#     f['fold_12'].create_dataset('input', data=x,
#                                 dtype='f4', chunks=(1, 191, 265, 2),
#                                 compression='lzf')
#     f['fold_12'].create_dataset('target', data=y,
#                                 dtype='f4', chunks=(1, 191, 265, 1),
#                                 compression='lzf')
#     f['fold_12'].create_dataset('patient_idx', data=pid)


# with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#     x = np.vstack(hf['fold_13']['input'][:])
#     y = np.vstack(hf['fold_13']['target'][:])
#     pid = np.repeat(hf['fold_13']['patient_idx'][:], 173)

# with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#     f['fold_13'].create_dataset('input', data=x,
#                                 dtype='f4', chunks=(1, 191, 265, 2),
#                                 compression='lzf')
#     f['fold_13'].create_dataset('target', data=y,
#                                 dtype='f4', chunks=(1, 191, 265, 1),
#                                 compression='lzf')
#     f['fold_13'].create_dataset('patient_idx', data=pid)


# with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#     x = np.vstack(hf['fold_10']['input'][:])
#     y = np.vstack(hf['fold_10']['target'][:])
#     pid = np.repeat(hf['fold_10']['patient_idx'][:], 173)

# with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#     f['fold_10'].create_dataset('input', data=x,
#                                 dtype='f4', chunks=(1, 191, 265, 2),
#                                 compression='lzf')
#     f['fold_10'].create_dataset('target', data=y,
#                                 dtype='f4', chunks=(1, 191, 265, 1),
#                                 compression='lzf')
#     f['fold_10'].create_dataset('patient_idx', data=pid)


# with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#     for i in range(10):
#         # del f[f'fold_{i}']
#         f.create_group(f'fold_{i}')

# with h5py.File('../../headneck_2d_new.h5', 'r') as f:
#     for i in range(14):
#         print(f[f'fold_{i}']['patient_idx'])
#         print((f[f'fold_{i}']['target'][:].sum(axis=(1, 2, 3)) > 0).sum())


def get_slice_idx(patients):
    idx = [0]
    for i, pid in enumerate(patients[1:]):
        if pid == patients[i]:
            idx.append(idx[-1] + 1)
        else:
            idx.append(0)
    return idx


# for i in range(14):
#     with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#         patients = f[f'fold_{i}']['patient_idx'][:]
#         slice_idx = get_slice_idx(patients)
#         f[f'fold_{i}'].create_dataset('slice_idx', data=slice_idx)


# for i in range(10):
#     print('fold', i)
#     with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#         x = np.vstack(hf[f'fold_{i}']['input'][:])
#         y = np.vstack(hf[f'fold_{i}']['target'][:])
#         pid = np.repeat(hf[f'fold_{i}']['patient_idx'][:], 173)

#     total = len(y)
#     filter_index = np.full(total, False)
#     for j in range(0, total, 173):
#         # positive slices per patient
#         positive_slice = np.sum(y[j:j+173].sum(axis=(1, 2, 3)) > 0)
#         print('p', j, 'pos_num', positive_slice)
#         step_num = max(int(np.rint((173-positive_slice)/positive_slice)), 1)
#         print('step', step_num)
#         filter_index[j:j+173:step_num] = True

#     filter_index[y.sum(axis=(1, 2, 3)) > 0] = True
#     filter_index[x.sum(axis=(1, 2, 3)) == 0] = False

#     with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#         f[f'fold_{i}'].create_dataset('input', data=x[filter_index],
#                                       dtype='f4', chunks=(1, 191, 265, 2),
#                                       compression='lzf')
#         f[f'fold_{i}'].create_dataset('target', data=y[filter_index],
#                                       dtype='f4', chunks=(1, 191, 265, 1),
#                                       compression='lzf')
#         f[f'fold_{i}'].create_dataset('patient_idx', data=pid[filter_index])

# for j in range(1, 5):
#     for i in range(10):
#         with h5py.File('../../headneck_3d_new.h5', 'r') as hf:
#             x = np.vstack(hf[f'fold_{i}']['input'][:])
#             y = np.vstack(hf[f'fold_{i}']['target'][:])
#             pid = np.repeat(hf[f'fold_{i}']['patient_idx'][:], 173)

#         filter_index = np.full(y.shape[0], False)
#         filter_index[j::5] = True
#         filter_index[y.sum(axis=(1, 2, 3)) > 0] = True
#         filter_index[x.sum(axis=(1, 2, 3)) == 0] = False

#         with h5py.File('../../headneck_2d_new.h5', 'a') as f:
#             f[f'fold_{(j+1)*10 + i}'].create_dataset(
#                 'input', data=x[filter_index],
#                 dtype='f4', chunks=(1, 191, 265, 2),
#                 compression='lzf')
#             f[f'fold_{(j+1)*10 + i}'].create_dataset(
#                 'target', data=y[filter_index],
#                 dtype='f4', chunks=(1, 191, 265, 1),
#                 compression='lzf')
#             f[f'fold_{(j+1)*10 + i}'].create_dataset(
#                 'patient_idx', data=pid[filter_index])
#             f[f'fold_{(j+1)*10 + i}'].create_dataset(
#                 'slice_idx', data=get_slice_idx(pid[filter_index]))


# with h5py.File(file_3d, 'r') as f:
#     x = f['fold_1']['input'][:]

# sns.boxplot(x=np.repeat([i for i in range(1, 6)], 173*191*265),
#             y=(x[:5][..., 0].flatten()-1024).clip(-100, 100))
# plt.show()

# x.shape

# x[..., 0].reshape((15, 173*191265))

# np.repeat([i for i in range(1, 5)], 173*191*265).shape
# x[:5][..., 0].flatten().shape

# np.quantile(x[0][..., 0], 0.25)
# plt.imshow(x[0][0][..., 0], 'gray', vmin=830)
# plt.show()

# plt.hist(x[0][0][..., 0][x[0][0][..., 0] > 24])
# plt.show()


# def _trisulf_data(image, threshold, color, opacity):
#     image = image.copy().transpose(2, 1, 0)
#     try:
#         verts, faces, normals, values = measure.marching_cubes(
#             image, threshold)
#         x, y, z = verts.T
#     except ValueError:
#         x, y, z = [0], [0], [0]
#         faces = [-1]
#         return None

#     fig = ff.create_trisurf(x=x, y=y, z=z,
#                             simplices=faces,
#                             plot_edges=False,
#                             show_colorbar=False,
#                             colormap=color,
#                             #  color_func=[color] * len(faces)
#                             )
#     data = fig['data'][0]
#     data.update(opacity=opacity)

#     return data


# with h5py.File(file_3d, 'r') as f:
#     pid = f['fold_10']['patient_idx'][:]


# for im, idx in enumerate(pid):
#     with h5py.File(file_3d, 'r') as f:
#         x = f['fold_10']['input'][im][..., 0]
#         target = f['fold_10']['target'][im][..., 0]

#     dummy = go.Scatter3d({'showlegend': False,
#                           'x': [], 'y': [], 'z': []
#                           })
#     fig = go.Figure(data=[
#         _trisulf_data(target, 0.5, 'rgb(23, 9, 92)', 0.5) or dummy,
#         # _trisulf_data(pred_mask_data, 0.5, 'rgb(255,0,0)', 0.5) or dummy,
#         _trisulf_data(x, 83, None, 0.1)
#     ])

#     steps = []
#     opacity = [data['opacity'] for data in fig['data']]
#     for i in range(10):
#         new_opacity = opacity.copy()
#         new_opacity[-1] = i*0.1
#         step = dict(
#             method="restyle",
#             args=[{"opacity": i*0.1}, [2]  # new_opacity}
#                   ],
#             label='{0:1.1f}'.format(i*0.1)
#         )
#         steps.append(step)

#     fig.update_layout(
#         title=f"Patient {idx:03d}",
#         sliders=[
#             go.layout.Slider(active=3,
#                              currentvalue={
#                                  "prefix": "Opacity: "},
#                              pad={"t": 50},
#                              len=500,
#                              lenmode='pixels',
#                              steps=steps,
#                              xanchor="right",
#                              ),
#         ],
#         updatemenus=[
#             go.layout.Updatemenu(
#                 type='buttons',
#                 active=0,
#                 pad={"r": 10, "t": 10},
#                 x=0.4,
#                 xanchor="left",
#                 buttons=[
#                     go.layout.updatemenu.Button(
#                         method='restyle',
#                         args=[{'visible': True}, [0]],
#                         args2=[{'visible': False}, [0]],
#                         label='Ground Truth'
#                     )]),
#             go.layout.Updatemenu(
#                 active=0,
#                 type='buttons',
#                 pad={"r": 10, "t": 10},
#                 x=0.4,
#                 xanchor="right",
#                 buttons=[
#                     go.layout.updatemenu.Button(
#                         method='restyle',
#                         args=[{'visible': True}, [1]],
#                         args2=[{'visible': False}, [1]],
#                         label='Prediction'
#                     )]
#             )]
#     )

#     html_file = f'../../val_images/{idx:03d}.html'

#     fig.write_html(html_file,
#                    auto_play=True,
#                    include_plotlyjs='cdn', include_mathjax='cdn')
