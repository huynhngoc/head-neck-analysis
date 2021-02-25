import os
import json
from itertools import product
from deoxys.utils import load_json_config

dirs = os.listdir('./templates')

base_template = {
    "train_params": {
        "epochs": 200,
        "callbacks": [
            {
                "class_name": "EarlyStopping",
                "config": {
                    "monitor": "val_loss",
                    "patience": 30
                }
            }
        ]
    }
}

base_path = 'templates/'
ds_params_path = base_path + 'dataset_params/'
preprocess_path = ds_params_path + 'options/'
inp_param_path = base_path + 'input_params/'
model_param_path = base_path + 'model_params/'
architecture_path = base_path + 'architecture/'


def generate_config(data, preprocess='CT_W_PET', aug=False,
                    patch_option=None,
                    model_params='adam_1e4',
                    architecture=''):
    # loading data reader
    if data not in ['2d', '3d', 'patch']:
        raise ValueError('Data must be one of "2d", "3d" or "patch"')
    data_reader = load_json_config(f'{ds_params_path}{data}.json')
    preprocess = load_json_config(f'{preprocess_path}{preprocess}.json')
    data_reader['config'].update(preprocess)
    if aug:
        data_aug = data if data != 'patch' else '3d'
        aug = load_json_config(f'{ds_params_path}{data_aug}/aug.json')
        data_reader['config'].update(aug)
    if patch_option and data == 'patch':
        p_options = load_json_config(
            f'{ds_params_path}{data}/{patch_option}.json')
        data_reader['config'].update(p_options)

    # load input params
    input_postfix = ''
    if patch_option:
        input_postfix = '_' + patch_option.split('_')[0]

    input_params = load_json_config(
        f'{inp_param_path}{data}{input_postfix}.json')

    # load model params
    model_params = load_json_config(
        f'{model_param_path}{model_params}.json')

    # load architecture
    if architecture:
        architecture = load_json_config(
            f'{architecture_path}{architecture}.json')

    config = base_template.copy()
    config['dataset_params'] = data_reader
    config['input_params'] = input_params
    config['model_params'] = model_params
    if architecture:
        config['architecture'] = architecture

    return config


def generate_multi_config(output_path='config/', **kwargs):
    params = []
    name_list = []
    arg_list = []
    for key, items in kwargs.items():
        params.append(key)
        if type(items) != list:
            name, val = items
            name_list.append([name])
            arg_list.append([val])
        else:
            name_list.append([item[0] for item in items])
            arg_list.append([item[1] for item in items])

    for name, args in zip(product(*name_list), product(*arg_list)):
        filename = '_'.join(name).replace('__', '_').strip('_')
        key_num = len(params)
        generate_config_params = {params[i]: args[i] for i in range(key_num)}
        content = generate_config(**generate_config_params)

        with open(f'{output_path}{filename}.json', 'w') as f:
            json.dump(content, f)


# generate_config('config.json', 'patch', aug=True, patch_option='80_bb')
generate_multi_config(output_path='config/2d/', data=[('2d', '2d')],
                      architecture=[('unet', '2d_unet_64_D4'),
                                    ('unet_32', '2d_unet_32_D4'),
                                    ('unet_48', '2d_unet_48_D4'),
                                    ('unet_32_D3', '2d_unet_32_D3'),
                                    ('unet_48_D3', '2d_unet_48_D3'),
                                    ('unet_64_D3', '2d_unet_64_D3'),
                                    ],
                      preprocess=[('', 'CT_W_PET'),
                                  ('norm', 'CT_W_PET_norm'),
                                  ('P10', 'CT_W_PET_C200_P10'),
                                  ('P25', 'CT_W_PET_C200_P25'),
                                  ],
                      aug=[('', False), ('aug', True)],
                      )

generate_multi_config(output_path='config/3d/', data=[('3d', '3d')],
                      architecture=[('unet_32', '3d_unet_32_D4'),
                                    ('unet_48', '3d_unet_48_D4'),
                                    ('unet_32_D3', '3d_unet_32_D3'),
                                    ('unet_48_D3', '3d_unet_48_D3'),
                                    ],
                      preprocess=[('', 'CT_W_PET'),
                                  ('norm', 'CT_W_PET_norm'),
                                  ('P10', 'CT_W_PET_C200_P10'),
                                  ('P25', 'CT_W_PET_C200_P25'),
                                  ],
                      aug=[('', False), ('aug', True)],
                      )
