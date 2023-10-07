import numpy as np
import os
from src.models.lstm import LSTMNN
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PATH_DATA = os.path.join("..", "..", "data", "processed")
PATH_TRAINING_DATA = os.path.join(PATH_DATA, "timeseries_20secpad_2023_train.npy")
PATH_VALIDATION_DATA = os.path.join(PATH_DATA, "timeseries_20secpad_2023_val.npy")
PATH_TEST_DATA = os.path.join(PATH_DATA, "timeseries_20sec_2023_val.npy")
PATH_MODEL = os.path.join("..", "..", "trained_models", "lstm_near_231003")
import site
import os

# Get the paths where Python is searching for installed packages
paths = site.getsitepackages()

# Check if TensorFlow is in any of these paths
for path in paths:
    if os.path.exists(os.path.join(path, 'tensorflow')):
        print(f'TensorFlow is installed in: {path}')
        break
if __name__ == "__main__":
    # mmap_mode allows for operations on the np.-array prior to loading into memory. So we can create the
    # indices and only load the data into memory in the randomized manner. We directly load the data
    # into memory in the way we want.
    data_train_mmap = np.load(PATH_TRAINING_DATA, mmap_mode='r')
    indices_train = np.random.choice(data_train_mmap.shape[0], data_train_mmap.shape[0]// 2, replace=False)
    data_train = data_train_mmap[indices_train].copy() # with copy we load the data into memory.
    data_train[np.isnan(data_train)] = -999.0
    data_val_mmap = np.load(PATH_VALIDATION_DATA, mmap_mode='r')
    indices_val = np.random.choice(data_val_mmap.shape[0], data_val_mmap.shape[0] // 10, replace=False)
    data_val = data_val_mmap[indices_val].copy()
    data_val[np.isnan(data_val)] = -999.0
    X_train = data_train[:, :, :-1] # last column is the time_to_arrival
    y_train = data_train[:, -1, -1] # the last y-value in the sequence is the target
    X_val = data_val[:, :, :-1]
    y_val = data_val[:, -1, -1]
    print("shape train", X_train.shape, "shape val", X_val.shape)
    print(X_train.dtype)
    n_features = X_train.shape[2]
    model = LSTMNN(n_features=n_features, use_masking=True, masking_value=-999.0)
    model.model.summary()
    patience_early = 1
    patience_reduce = 1
    reduce_factor = 0.7
    batch_size = 512
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        patience_early=patience_early,
        patience_reduce=patience_reduce,
        reduce_factor=reduce_factor,
        batch_size=batch_size,
    )

    data_test = np.load(PATH_TEST_DATA)
    data_test[np.isnan(data_test)] = -999.0
    X_test = data_test[:, :, :-1]
    y_test = data_test[:, -1, -1]
    print("shape test", X_test.shape)
    model.evaluate(X_test, y_test)
