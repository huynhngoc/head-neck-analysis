import json
se_norm_block = '''{{
        "name": "SE_block_{id}_in",
        "class_name": "Conv3D",
        "config": {{
            "filters": {filter},
            "kernel_size": {kernel},
            "activation": "relu",
            "kernel_initializer": "he_normal",
            "padding": "same"
        }}
    }},
    {{
        "name": "SE_block_{id}_norm",
        "class_name": "InstanceNormalization",
        "inputs": [
            "SE_block_{id}_in"
        ]
    }},
    {{
        "class_name": "GlobalAvgPool3D",
        "inputs": [
            "SE_block_{id}_in"
        ]
    }},
    {{
        "class_name": "Reshape",
        "config": {{
            "target_shape": [
                1,
                1,
                1,
                {filter}
            ]
        }}
    }},
    {{
        "class_name": "Dense",
        "config": {{
            "units": {units},
            "activation": "relu"
        }}
    }},
    {{
        "name": "SE_block_{id}_gamma",
        "class_name": "Dense",
        "config": {{
            "units": {filter},
            "activation": "sigmoid"
        }}
    }},
    {{
        "class_name": "GlobalAvgPool3D",
        "inputs": [
            "SE_block_{id}_in"
        ]
    }},
    {{
        "class_name": "Reshape",
        "config": {{
            "target_shape": [
                1,
                1,
                1,
                {filter}
            ]
        }}
    }},
    {{
        "class_name": "Dense",
        "config": {{
            "units": {units},
            "activation": "relu"
        }},
        "normalizer": {{
            "class_name": "BatchNormalization"
        }}
    }},
    {{
        "name": "SE_block_{id}_beta",
        "class_name": "Dense",
        "config": {{
            "units": {filter},
            "activation": "tanh"
        }}
    }},
    {{
        "name": "SE_block_{id}_preout",
        "class_name": "Multiply",
        "inputs": [
            "SE_block_{id}_in",
            "SE_block_{id}_gamma"
        ]
    }},
    {{
        "name": "SE_block_{id}_out",
        "class_name": "Add",
        "inputs": [
            "SE_block_{id}_preout",
            "SE_block_{id}_beta"
        ]
}},'''


def conv_se_block(id, n_filter, reduction=2, kernel_size=3, input_name=None):
    if input_name is None:
        conv_block_template = '''{{
            "name": "conv_{id}_in",
            "class_name": "Conv3D",
            "config": {{
                "filters": {filter},
                "kernel_size": {kernel},
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "padding": "same"
            }}
        }},'''
        conv_block = conv_block_template.format(
            id=id, filter=n_filter, kernel=kernel_size)
    else:
        conv_block_template = '''{{
            "name": "conv_{id}_in",
            "class_name": "Conv3D",
            "config": {{
                "filters": {filter},
                "kernel_size": {kernel},
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "padding": "same"
            }},
            "inputs":["{input_name}"]
        }},'''
        conv_block = conv_block_template.format(
            id=id, filter=n_filter, kernel=kernel_size, input_name=input_name)

    se_norm = se_norm_block.format(
        id=id,
        filter=n_filter, units=n_filter // reduction, kernel=kernel_size)

    return conv_block + se_norm


def res_se_block(input_name, id, n_filter, reduction=2, kernel_size=3):
    res_se = conv_se_block(
        id, n_filter, reduction=reduction, kernel_size=kernel_size)
    res_block_template = '''{{
        "name": "conv_{id}_out",
        "class_name": "Add",
        "inputs":["{input_name}", "{conv_se_block_name}"]
    }},'''
    res_block = res_block_template.format(
        id=id, input_name=input_name,
        conv_se_block_name=f'SE_block_{id}_out')
    return res_se + res_block


def res_se_block_identity(input_name, id, n_filter, reduction=2, kernel_size=3):
    res_se_1 = conv_se_block(
        f'{id}_1', n_filter, reduction=reduction,
        kernel_size=3, input_name=input_name)
    res_se_2 = conv_se_block(
        f'{id}_2', n_filter, reduction=reduction,
        kernel_size=1, input_name=input_name)
    res_block_template = '''{{
        "name": "conv_{id}_out",
        "class_name": "Add",
        "inputs":["{conv_se_block_name}", "{conv_se_block_name_2}"]
    }},'''
    res_block = res_block_template.format(
        id=id, input_name=input_name,
        conv_se_block_name=f'SE_block_{id}_1_out',
        conv_se_block_name_2=f'SE_block_{id}_2_out')
    return res_se_1 + res_se_2 + res_block


def gen_se_model():
    final_config = ''
    n_filter = 24
    reduction = 2

    max_pool_block = '''{{
        "name":"maxpool_{}",
        "class_name":"MaxPooling3D"
    }},
    '''
    conv_trans_block = '''{{
        "name": "upconv_{id}",
        "class_name":"Conv3DTranspose",
        "config": {{
            "filters": {filter},
            "kernel_size": 3,
            "strides": 2,
            "kernel_initializer": "he_normal",
            "padding": "same"
        }}
    }},
    '''
    left_1_1 = res_se_block_identity(
        input_name='input_1', id='1',
        n_filter=n_filter, reduction=reduction, kernel_size=7)
    # output = conv_1_out
    left_1_2 = res_se_block(
        input_name='conv_1_out', id='2',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_2_out
    max_pool_1 = max_pool_block.format(1)
    # output = maxpool_1
    final_config += left_1_1 + left_1_2 + max_pool_1

    n_filter *= 2
    left_2_1 = res_se_block_identity(
        input_name='maxpool_1', id='3',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_3_out
    left_2_2 = res_se_block(
        input_name='conv_3_out', id='4',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_4_out
    left_2_3 = res_se_block(
        input_name='conv_4_out', id='5',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_5_out

    max_pool_2 = max_pool_block.format(2)
    # output = maxpool_2
    final_config += left_2_1 + left_2_2 + left_2_3 + max_pool_2

    n_filter *= 2
    left_3_1 = res_se_block_identity(
        input_name='maxpool_2', id='6',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_6_out
    left_3_2 = res_se_block(
        input_name='conv_6_out', id='7',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_7_out
    left_3_3 = res_se_block(
        input_name='conv_7_out', id='8',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_8_out

    max_pool_3 = max_pool_block.format(3)
    # output = maxpool_3
    final_config += left_3_1 + left_3_2 + left_3_3 + max_pool_3

    n_filter *= 2
    left_4_1 = res_se_block_identity(
        input_name='maxpool_3', id='9',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_9_out
    left_4_2 = res_se_block(
        input_name='conv_9_out', id='10',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_10_out
    left_4_3 = res_se_block(
        input_name='conv_10_out', id='11',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_11_out

    max_pool_4 = max_pool_block.format(4)
    # output = maxpool_4
    final_config += left_4_1 + left_4_2 + left_4_3 + max_pool_4

    n_filter *= 2
    left_5_1 = res_se_block_identity(
        input_name='maxpool_4', id='12',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_12_out
    left_5_2 = res_se_block(
        input_name='conv_12_out', id='13',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_13_out
    left_5_3 = res_se_block(
        input_name='conv_13_out', id='14',
        n_filter=n_filter, reduction=reduction, kernel_size=3)
    # output = conv_14_out
    final_config += left_5_1 + left_5_2 + left_5_3

    n_filter /= 2
    upconv_1 = conv_trans_block.format(id=1, filter=n_filter)
    # output = upconv_1
    right_4_1 = conv_se_block(id='15', n_filter=n_filter, reduction=reduction,
                              kernel_size=3, input_name='upconv_1","conv_11_out'
                              )
    # output = SE_block_15_out
    right_4_2 = conv_se_block(id='16', n_filter=n_filter, reduction=reduction,
                              kernel_size=3)
    # output = SE_block_16_out

    final_config += upconv_1 + right_4_1 + right_4_2

    n_filter /= 2
    upconv_2 = conv_trans_block.format(id=2, filter=n_filter)
    # output = upconv_2
    right_3_1 = conv_se_block(id='17', n_filter=n_filter, reduction=reduction,
                              kernel_size=3, input_name='upconv_2","conv_8_out'
                              )
    # output = SE_block_17_out
    right_3_2 = conv_se_block(id='18', n_filter=n_filter, reduction=reduction,
                              kernel_size=3)
    # output = SE_block_18_out

    final_config += upconv_2 + right_3_1 + right_3_2

    n_filter /= 2
    upconv_3 = conv_trans_block.format(id=3, filter=n_filter)
    # output = upconv_3
    right_2_1 = conv_se_block(id='19', n_filter=n_filter, reduction=reduction,
                              kernel_size=3, input_name='upconv_3","conv_5_out'
                              )
    # output = SE_block_19_out
    right_2_2 = conv_se_block(id='20', n_filter=n_filter, reduction=reduction,
                              kernel_size=3)
    # output = SE_block_20_out

    final_config += upconv_3 + right_2_1 + right_2_2

    n_filter /= 2
    upconv_4 = conv_trans_block.format(id=4, filter=n_filter)
    # output = upconv_4
    right_1_1 = conv_se_block(id='21', n_filter=n_filter, reduction=reduction,
                              kernel_size=3, input_name='upconv_4","conv_2_out'
                              )
    # output = SE_block_21_out

    final_config += upconv_4 + right_1_1

    final_config += conv_se_block(
        id=22, n_filter=n_filter, reduction=reduction,
        kernel_size=1, input_name='SE_block_20_out')
    final_config += conv_se_block(
        id=23, n_filter=n_filter, reduction=reduction,
        kernel_size=1, input_name='SE_block_18_out')
    final_config += conv_se_block(
        id=24, n_filter=n_filter, reduction=reduction,
        kernel_size=1, input_name='SE_block_16_out')

    final_config += '''{
        "class_name": "AddResize",
        "inputs": ["SE_block_21_out", "SE_block_22_out", "SE_block_23_out", "SE_block_24_out"]
    },
    '''

    right_1_2 = conv_se_block(id='25', n_filter=n_filter, reduction=reduction,
                              kernel_size=3)
    # output = SE_block_25_out
    final_config += right_1_2
    final_config += '''
    {
        "class_name": "Conv3D",
        "config": {
            "filters": 1,
            "kernel_size": 1,
            "activation": "sigmoid",
            "kernel_initializer": "he_normal",
            "padding": "same"
        }
    }
    '''

    return final_config


gen_se_model()
with open("SE_model.json", 'w') as f:
    f.write(gen_se_model())
