import copy
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import KFold
import sys
import os
from src.models.linear_model import LinearModel
from src.processing_utils.preprocessing import seconds_till_arrival

DIR_TRAINING_DATA = os.path.join("..", "..", "data", "final", "train")
DIR_TEST_DATA = os.path.join("..", "..", "data", "final", "test")

FILE_TRAINING_DATA = os.path.join(DIR_TRAINING_DATA,"training_data_2022_100km.csv")
FILE_TEST_DATA = os.path.join(DIR_TEST_DATA,"testdata_2023_100km_comparable.csv")

PATH_STD_SCALER = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_100km.bin")
PATH_MINMAX_SCALER = os.path.join("..", "..", "trained_models", "scalers", "minmax_scaler_100km_h3.bin")
PATH_MODEL =os.path.join("..", "..", "trained_models", "best_models")
PATH_LOG_FILE = os.path.join("..", "..", "trained_models", "best_models", "logs", "polynomial_linear_model_100km.txt")

COLS_TO_SCALE_STD = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

COLS_TO_SCALE_MINMAX = ["density_10_minutes_past", "density_30_minutes_past", "density_60_minutes_past"]

FEATURES = [
    'distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday',
    'sec_sin', 'sec_cos', 'day_sin', 'day_cos', 'bearing_sin', 'bearing_cos',
    'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'density_10_minutes_past', 'density_30_minutes_past', 'density_60_minutes_past'
]


if __name__ == "__main__":
    std_scaler = load(PATH_STD_SCALER)
    minmax_scaler = load(PATH_MINMAX_SCALER)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    best_mae = 10000
    df_train = pd.read_csv(FILE_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_test = pd.read_csv(FILE_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_train = seconds_till_arrival(df_train)/df_train.distance
    y_test = seconds_till_arrival(df_test)/df_test.distance
    for i in range(1, 20):
        degrees = i
        model = LinearModel(
            std_scaler=std_scaler,
            minmax_scaler=minmax_scaler,
            features=FEATURES,
            pol_degree=degrees,
            cols_to_scale_std=COLS_TO_SCALE_STD,
            cols_to_scale_minmax=COLS_TO_SCALE_MINMAX
        )

        fold_maes = []
        fold_r2s = []

        for train_index, val_index in kf.split(df_train):
            train_fold, val_fold = df_train.iloc[train_index], df_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model.fit(train_fold, y_train_fold)
            mae, r2 = model.evaluate(val_fold, y_val_fold)

            fold_maes.append(mae)
            fold_r2s.append(r2)

        avg_mae = np.mean(fold_maes)
        avg_r2 = np.mean(fold_r2s)

        print("degree", degrees, "avg mae", avg_mae, "avg r2", avg_r2)
        mae, r2 = model.evaluate(df_test, y_test)
        print("degree", degrees, "test mae", mae, "test r2", r2)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_model = copy.deepcopy(model.model)

        else:
            break

    model_file = os.path.join(PATH_MODEL, "polynomial_regression_100km_rel" + str(degrees -1) + ".sav")
    dump(best_model, model_file)
    """
    doesn't work yet for some reason
    model = LinearModel(
        model_file=model_file,
        features=FEATURES,
        pol_degree=degrees -1,
        cols_to_scale_std=COLS_TO_SCALE_STD,
        cols_to_scale_minmax=COLS_TO_SCALE_MINMAX
    )

    mae, r2 = model.evaluate(df_test, y_test)
    print("degree", degrees, "test mae", mae, "test r2", r2)
    """

