from deoxys.experiment import ExperimentPipeline
import numpy as np
import argparse
import os
import shutil
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
import customize_obj

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--initial_epoch", default=200, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--best_epoch", default=0, type=int)
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx,slice_idx', type=str)
    parser.add_argument("--monitor", default='', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    if 'patch' in args.log_folder:
        analysis_folder = args.analysis_folder
    else:
        analysis_folder = ''

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    # copy to another location
    log_folder = args.log_folder + '_' + args.dataset_file[:-5].split('/')[-1]
    if not os.path.exists(log_folder):
        shutil.copytree(args.log_folder, log_folder)

    ex = ExperimentPipeline(
        log_base_path=log_folder,
        temp_base_path=args.temp_folder + '_' +
        args.dataset_file[:-5].split('/')[-1]
    )
    if args.best_epoch == 0:
        try:
            ex = ex.load_best_model(
                recipe='auto',
                analysis_base_path=analysis_folder,
                map_meta_data=meta,
            )
        except Exception as e:
            print("Error while loading best model", e)
            print(e)
    else:
        print(f'Loading model from epoch {args.best_epoch}')
        ex.from_file(args.log_folder +
                     f'/model/model.{args.best_epoch:03d}.h5')
    weights = ex.model._model.optimizer.get_weights()
    weights[0] = np.array(args.initial_epoch *
                          os.environ.get('ITER_PER_EPOCH', 200))
    ex.model._model.optimizer.set_weights(weights)

    print('Optimizer state:', ex.model._model.optimizer.iterations)
    print('original learning_rate:', ex.model._model.optimizer.learning_rate)
    ex.load_new_dataset(
        args.dataset_file,
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs+args.initial_epoch,
        initial_epoch=args.initial_epoch
    ).apply_post_processors(
        recipe='auto',
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
        run_test=True
    ).plot_3d_test_images(best_num=2, worst_num=2)

    ex.run_test().apply_post_processors(
        recipe='auto',
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
        run_test=True
    ).plot_3d_test_images(best_num=2, worst_num=2)
