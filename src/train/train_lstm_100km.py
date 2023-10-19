import numpy as np
import os
from joblib import load as load_joblib
from src.models.lstm import LSTMNN

PATH_DATA = os.path.join("..", "..", "data", "processed")
PATH_TRAINING_DATA = os.path.join(PATH_DATA, "timeseries_10sec_2022_100km_train.npy")
PATH_VALIDATION_DATA = os.path.join(PATH_DATA, "timeseries_10sec_2022_100km_val.npy")
PATH_TEST_DATA = os.path.join(PATH_DATA, "testdata_2023_100km_comparable.npy")
PATH_MODEL = os.path.join("..", "..", "trained_models", "lstm_near_231003")
scaler_path_std = os.path.join("../..", "trained_models", "std_scaler_100km_h3.bin")

scaler = load_joblib(scaler_path_std)
if __name__ == "__main__":

    data_train = np.load(PATH_TRAINING_DATA)
    data_val = np.load(PATH_VALIDATION_DATA)

    X_train = data_train[:, :, :-1] # last column is the time_to_arrival
    y_train = data_train[:, -1, -1] # the last y-value in the sequence is the target
    X_val = data_val[:, :, :-1]
    y_val = data_val[:, -1, -1]
    print("shape train", X_train.shape, "shape val", X_val.shape)
    n_features = X_train.shape[2]

    model = LSTMNN(
        scaler=scaler,
        distance_relative=True,
        index_distance=0,
        n_features=n_features,
        lr=0.001,
        lstm_layers=(256,),
        dense_layers=(512, 256),
        dropout_rate_fc=0.2,
        dropout_rate_lstm=0.2,
    )



    patience_early = 2
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
    X_test = data_test[:, :, :-1]
    y_test = data_test[:, -1, -1]
    print("shape test", X_test.shape)
    model.evaluate(X_test, y_test)
