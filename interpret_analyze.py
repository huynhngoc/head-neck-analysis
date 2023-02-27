import numpy as np
import h5py
import argparse
import pandas as pd


def avg_filter(data):
    return np.concatenate([
        [data],  # (0,0,0)
        [np.roll(data, 1, axis=i) for i in range(3)],  # (one 1)
        [np.roll(data, -1, axis=i) for i in range(3)],
        [np.roll(data, 1, axis=p) for p in [(0, 1), (0, 2), (1, 2)]],  # two 1s
        [np.roll(data, -1, p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, (-1, 1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, (1, -1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, r, (0, 1, 2)) for r in [
                (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
            1, -1]
         ]
    ]).mean(axis=0)


def edge_detection(data):
    data_neg = 0 - data
    return np.concatenate([
        [data] * 26,  # (0,0,0)
        [np.roll(data_neg, 1, axis=i) for i in range(3)],  # (one 1)
        [np.roll(data_neg, -1, axis=i) for i in range(3)],
        [np.roll(data_neg, 1, axis=p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, -1, p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, (-1, 1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, (1, -1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, r, (0, 1, 2)) for r in [
                (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
            1, -1]
         ]
    ]).mean(axis=0)


def get_overall_info(data):
    print('Getting basic statistical information')
    return {
        'ct_total': (data[..., 0] > 0).sum(),
        'ct_sum': data[..., 0].sum(),
        'ct_max': data[..., 0].max(),
        'ct_mean': data[..., 0].mean(),
        'ct_std': data[..., 0].std(),
        'ct_q1': np.quantile(data[..., 0], 0.25),
        'ct_q2': np.quantile(data[..., 0], 0.5),
        'ct_q3': np.quantile(data[..., 0], 0.75),
        'pt_total': (data[..., 1] > 0).sum(),
        'pt_sum': data[..., 1].sum(),
        'pt_max': data[..., 1].max(),
        'pt_mean': data[..., 1].mean(),
        'pt_std': data[..., 1].std(),
        'pt_q1': np.quantile(data[..., 1], 0.25),
        'pt_q2': np.quantile(data[..., 1], 0.5),
        'pt_q3': np.quantile(data[..., 1], 0.75),
    }


def get_area_info(data, area, name):
    print('Getting information based information of', name)
    selected_data = data[area > 0]
    return {
        f'ct_{name}_total': (selected_data[..., 0] > 0).sum(),
        f'ct_{name}_sum': selected_data[..., 0].sum(),
        f'ct_{name}_max': selected_data[..., 0].max(),
        f'ct_{name}_mean': selected_data[..., 0].mean(),
        f'ct_{name}_std': selected_data[..., 0].std(),
        f'ct_{name}_q1': np.quantile(selected_data[..., 0], 0.25),
        f'ct_{name}_q2': np.quantile(selected_data[..., 0], 0.5),
        f'ct_{name}_q3': np.quantile(selected_data[..., 0], 0.75),
        f'pt_{name}_total': (selected_data[..., 1] > 0).sum(),
        f'pt_{name}_sum': selected_data[..., 1].sum(),
        f'pt_{name}_max': selected_data[..., 1].max(),
        f'pt_{name}_mean': selected_data[..., 1].mean(),
        f'pt_{name}_std': selected_data[..., 1].std(),
        f'pt_{name}_q1': np.quantile(selected_data[..., 1], 0.25),
        f'pt_{name}_q2': np.quantile(selected_data[..., 1], 0.5),
        f'pt_{name}_q3': np.quantile(selected_data[..., 1], 0.75),
    }


def get_histogram_info(data, areas, names):
    print('Getting distribution data of', names)
    objs = {}
    for (area, name) in zip(areas, names):
        selected_data = data[area > 0]
        objs.update({
            f'{name}_total': (selected_data > 0).sum(),
            f'{name}_max': selected_data.max(),
            f'{name}_mean': selected_data.mean(),
            f'{name}_std': selected_data.std(),
            f'{name}_median': np.median(selected_data),
        })


def get_info(data_normalized, ct_img, pt_img, tumor, node):
    # start the count
    overall_info = get_overall_info(data_normalized)

    # vargrad in tumour & node
    tumor_info = get_area_info(data_normalized, tumor, 'tumor')
    node_info = get_area_info(data_normalized, node, 'node')
    normal_voxel_info = get_area_info(data_normalized,
                                      1 - tumor - node,
                                      'outside')
    # correlation between intensity and vargrad
    suv_corr = np.corrcoef(
        pt_img.flatten(), data_normalized[..., 1].flatten())[0, 1]
    hu_corr = np.corrcoef(
        ct_img.flatten(), data_normalized[..., 0].flatten())[0, 1]

    # histogram data
    suv_0_2 = (pt_img <= 0.08).astype(int)
    suv_2_4 = (pt_img <= 0.16).astype(int) - suv_0_2
    suv_4_6 = (pt_img <= 0.24).astype(int) - suv_0_2 - suv_2_4
    suv_6_8 = (pt_img <= 0.32).astype(int) - suv_0_2 - suv_2_4 - suv_4_6
    suv_8_10 = (pt_img <= 0.4).astype(int) - \
        suv_0_2 - suv_2_4 - suv_4_6 - suv_6_8
    suv_10_over = (pt_img > 0.4).astype(int)

    areas = [suv_0_2, suv_2_4, suv_4_6, suv_6_8, suv_8_10, suv_10_over]
    area_names = ['suv_0_2', 'suv_2_4', 'suv_4_6',
                  'suv_6_8', 'suv_8_10', 'suv_10_over']
    suv_info = get_histogram_info(data_normalized, areas, area_names)

    all_info = {
        **overall_info,
        **tumor_info,
        **node_info,
        **normal_voxel_info,
        'hu_corr': hu_corr,
        'suv_corr': suv_corr,
        **suv_info
    }

    return all_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file")
    parser.add_argument("log_folder")
    parser.add_argument("--idx", default=0, type=int)

    args, unknown = parser.parse_known_args()

    base_folder = args.log_folder
    h5_file = args.h5_file

    print(f'Checking folder {base_folder}')

    if 'ous' in h5_file:
        center = 'OUS'
        print('Getting OUS dataset file...')
        src_data_file = '/mnt/project/ngoc/datasets/headneck/outcome_node_new.h5'
        curr_fold = f'fold_{base_folder[-1]}'
        with h5py.File(src_data_file, 'r') as f:
            pids = f[curr_fold]['patient_idx'][:]
    elif 'maastro' in h5_file:
        center = 'MAASTRO'
        src_data_file = '/mnt/project/ngoc/datasets/headneck/outcome_maastro_node.h5'
        with h5py.File(src_data_file, 'r') as f:
            print('Getting MAASTRO dataset file')
            pids = []
            for key in f.keys():
                pids.extends([
                    pid for pid in f[key]['patient_idx'][:]
                ])
                if len(pids) > args.idx:
                    curr_fold = key
                    break
    print('List of PIDS:', pids)
    try:
        pid = pids[args.idx]
        print('Patients:', pid)
        print('Getting image data...')
        with h5py.File(src_data_file, 'r') as f:
            img_idx = [pid
                       for pid in f[curr_fold]['patient_idx'][:]].index(pid)
            img = f[curr_fold]['image'][img_idx]
            dfs = f[curr_fold]['DFS'][img_idx]
            os = f[curr_fold]['OS'][img_idx]

        print('Windowing CT images')
        # windowing
        ct_img = img[..., 0] - (1024 + 70)
        ct_img = ((ct_img.clip(-100, 100) + 100) / 200).clip(0, 1)

        print('Normalize PET images')
        # clipped SUV
        pt_img = (img[..., 1] / 25).clip(0, 1)

        tumor = img[..., 2]
        node = img[..., 3]

        print('Getting interpret resutls...')
        with h5py.File(args.log_folder + '/' + h5_file, 'r') as f:
            data = f[pid][:]

        thres = np.quantile(data, 0.99)
        max_vargrad = data.max()

        print('Normalizing interpret results...')
        data_normalized = ((data - thres) / (max_vargrad - thres)).clip([0, 1])

        info_raw = get_info(data_normalized, ct_img, pt_img, tumor, node)
        raw_df = pd.DataFrame(info_raw)
        raw_df.insert(0, 'vargrad_threshold', thres)
        raw_df.insert(0, 'vargrad_max', max_vargrad)
        raw_df.insert(0, 'os', os)
        raw_df.insert(0, 'dfs', dfs)
        raw_df.insert(0, 'center', center)
        raw_df.insert(0, 'pid', pid)

        print('Saving raw resutls...')
        raw_df.to_csv(
            base_folder + f'/{center}/raw_interpret_{pid}.csv', index=False)

        print('Smoothening interpret results...')
        smoothen_data = avg_filter(data)
        s_thres = np.quantile(smoothen_data, 0.99)
        s_max_vargrad = smoothen_data.max()

        print('Normalizing smoothen interpret results...')
        s_data_normalized = ((smoothen_data - s_thres) /
                             (s_max_vargrad - s_thres)).clip([0, 1])
        info_smooth = get_info(s_data_normalized, ct_img, pt_img, tumor, node)
        smoothen_df = pd.DataFrame(info_smooth)
        smoothen_df.insert(0, 'vargrad_threshold', s_thres)
        smoothen_df.insert(0, 'vargrad_max', s_max_vargrad)
        smoothen_df.insert(0, 'os', os)
        smoothen_df.insert(0, 'dfs', dfs)
        smoothen_df.insert(0, 'center', center)
        smoothen_df.insert(0, 'pid', pid)

        print('Saving smoothen results...')
        smoothen_df.to_csv(
            base_folder + f'/{center}/raw_interpret_{pid}.csv', index=False)

    except Exception:
        print('Index not found!! Exiting')
