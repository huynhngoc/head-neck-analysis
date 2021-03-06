# from deoxys.experiment import Experiment
# from deoxys.utils import read_file
import argparse
from customize_obj import PostProcessor
# import numpy as np
# import h5py
# import pandas as pd
# import os

# from deoxys.utils import load_json_config
# from deoxys.loaders import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder",
                        default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--meta", default='patient_idx,slice_idx', type=str)
    parser.add_argument("--monitor", default='', type=str)

    args, unknown = parser.parse_known_args()

    if '2d' in args.log_folder:
        PostProcessor(
            args.log_folder,
            temp_base_path=args.temp_folder,
            map_meta_data=args.meta
        ).map_2d_meta_data().calculate_fscore_single().merge_2d_slice(
        ).calculate_fscore().get_best_model(args.monitor)
    elif 'patch' in args.log_folder:
        PostProcessor(
            args.log_folder,
            temp_base_path=args.temp_folder,
            analysis_base_path=args.analysis_folder,
            map_meta_data=args.meta
        ).merge_3d_patches(
        ).calculate_fscore().get_best_model(args.monitor)
    else:
        PostProcessor(
            args.log_folder,
            temp_base_path=args.temp_folder,
            map_meta_data=args.meta.split(',')[0]
        ).map_2d_meta_data(
        ).calculate_fscore_single_3d().get_best_model(args.monitor)
