import numpy as np
import os
from joblib import load as load_joblib
from src.models.lstm import LSTMNN

PATH_DATA_TRAIN = os.path.join("..", "..", "data", "final", "train")
PATH_DATA_TEST = os.path.join("..", "..", "data", "final", "test")
PATH_TRAINING_DATA = os.path.join(PATH_DATA_TRAIN, "timeseries_30sec_2022_all_distances_train_clean.npy")
PATH_VALIDATION_DATA = os.path.join(PATH_DATA_TRAIN, "timeseries_30sec_2022_all_distances_optim_clean.npy")
PATH_TEST_DATA = os.path.join(PATH_DATA_TEST, "testdata_2023_comparable_clean.npy")

PATH_MODEL ="C:/Users/dario/Documents/Master Data Science/Lufthansa/Lufthansa-Arrival-Time-Prediction/trained_models/best_models/lstm_full_distance"

PATH_STD_SCALER = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_all_distances.bin")
std_scaler = load_joblib(PATH_STD_SCALER)

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
        scaler=std_scaler,
        distance_relative=True,
        index_distance=0,
        n_features=n_features,
        lr=0.00016,
        lstm_layers=(1024,),
        dense_layers=(3492, 3666),
        dropout_rate_fc=0.510,
        dropout_rate_lstm=0.05,
    )



    patience_early = 8
    patience_reduce = 5
    reduce_factor = 0.7
    batch_size = 298
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        patience_early=patience_early,
        patience_reduce=patience_reduce,
        reduce_factor=reduce_factor,
        batch_size=batch_size,
        max_epochs=500
    )

    data_test = np.load(PATH_TEST_DATA)
    X_test = data_test[:, :, :-1]
    y_test = data_test[:, -1, -1]
    print("shape test", X_test.shape)
    model.evaluate(X_test, y_test)
    model.model.save(PATH_MODEL)


