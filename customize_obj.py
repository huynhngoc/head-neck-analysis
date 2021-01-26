from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator

from deoxys.customize import custom_datareader


class H5PatchReader:
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold',
                 patch_size=128, overlap=0.5, shuffle=False,
                 stratify=False, augmentations=False, postprocessors=None):
        pass
