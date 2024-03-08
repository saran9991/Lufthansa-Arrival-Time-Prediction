import numpy as np
import os
from joblib import load as load_joblib
from src.models.lstm import LSTMNN

PATH_DATA_TRAIN = os.path.join("..", "..", "data", "final", "train")
PATH_DATA_TEST = os.path.join("..", "..", "data", "final", "test")
PATH_TRAINING_DATA = os.path.join(PATH_DATA_TRAIN, "timeseries_10sec_2022_100km_train_clean.npy")
PATH_VALIDATION_DATA = os.path.join(PATH_DATA_TRAIN, "timeseries_10sec_2022_100km_optim_clean.npy")
PATH_TEST_DATA = os.path.join(PATH_DATA_TEST, "testdata_2023_100km_comparable_clean.npy")

PATH_MODEL =".." + os.sep + ".." + os.sep + "trained_models" + os.sep + "best_models" + os.sep + "lstm_100km"

PATH_STD_SCALER = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_100km.bin")
std_scaler = load_joblib(PATH_STD_SCALER)


SHAP_FEATURES = [
    'Distance to Airport',
    'Altitude',
    'Geoaltitude',
    'Vertical Rate',
    'Groundspeed',
    'Holiday',
    'Sine Second in Day',
    'Cosine Second in Day',
    'Sine Day in Year',
    'Cosine Day in Year',
    'Bearing Sine',
    'Bearing Cosine',
    'Track Sine',
    'Track Cosine',
    'Latitude Radians',
    'Longitude Radians',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
    "H3 Density 10 min",
    "H3 Density 30 min",
    "H3 Density 60 min",
    "Seconds Till Last Timestamp"
]
if __name__ == "__main__":
    data_train = np.load(PATH_TRAINING_DATA)
    data_val = np.load(PATH_VALIDATION_DATA)
    # Define the number of samples you want
    sample_size = 1000

    # Ensure the dataset has enough samples
    if data_train.shape[0] < sample_size or data_val.shape[0] < sample_size:
        raise ValueError("The dataset size is smaller than the requested sample size.")

    # Randomly sample indices
    sample_indices_train = np.random.choice(data_train.shape[0], size=sample_size, replace=False)
    sample_indices_val = np.random.choice(data_val.shape[0], size=sample_size, replace=False)

    # Subset the data
    data_train_sampled = data_train[sample_indices_train]
    data_val_sampled = data_val[sample_indices_val]

    # Now get X and y for both training and validation
    X_train = data_train_sampled[:, :, :-1]
    y_train = data_train_sampled[:, -1, -1]

    X_val = data_val_sampled[:, :, :-1]
    y_val = data_val_sampled[:, -1, -1]
    print("shape train", X_train.shape, "shape val", X_val.shape)
    n_features = X_train.shape[2]

    model = LSTMNN(
        scaler=std_scaler,
        distance_relative=True,
        index_distance=0,
        n_features=n_features,
        lr=0.0001,
        lstm_layers=(813, 281),
        dense_layers=(1425, 4096, 588),
        dropout_rate_fc=0.2,
        dropout_rate_lstm= 0.1235,
        model_file=PATH_MODEL
    )

    SHAP_FEATURES = [
        'Distance to Airport',
        'Altitude',
        'Geoaltitude',
        'Vertical Rate',
        'Groundspeed',
        "H3 Density 10 min",
        "H3 Density 30 min",
        "H3 Density 30 min",
        'Holiday',
        'Sine Second in Day',
        'Cosine Second in Day',
        'Sine Day in Year',
        'Cosine Day in Year',
        'Bearing Sine',
        'Bearing Cosine',
        'Track Sine',
        'Track Cosine',
        'Latitude Radians',
        'Longitude Radians',
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday',
        "seconds till last step"
    ]
    model.get_shap(X_train, X_val, FEATURES= SHAP_FEATURES, file= "shap_plot_lstm.png")
