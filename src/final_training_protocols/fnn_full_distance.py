import numpy as np
import pandas as pd
from joblib import dump, load
import os
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival


DIR_TRAINING_DATA = os.path.join("..", "..", "data", "final", "train")
DIR_TEST_DATA = os.path.join("..", "..", "data", "final", "test")

FILE_TRAINING_DATA = os.path.join(DIR_TRAINING_DATA,"training_data_whole.csv")
FILE_TEST_DATA = os.path.join(DIR_TEST_DATA,"testdata_2023_comparable.csv")

PATH_STD_SCALER = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_all_distances.bin")
PATH_MINMAX_SCALER = os.path.join("..", "..", "trained_models", "scalers", "minmax_scaler_all_distances_h3.bin")
PATH_MODEL ="C:/Users/dario/Documents/Master Data Science/Lufthansa/Lufthansa-Arrival-Time-Prediction/trained_models/best_models/fnn_full_distance"

COLS_TO_SCALE_STD = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

COLS_TO_SCALE_MINMAX = ["density_10_minutes_past", "density_30_minutes_past", "density_60_minutes_past"]


FEATURES = [
    'distance',
    'altitude',
    'geoaltitude',
    'vertical_rate',
    'groundspeed',
    'holiday',
    'sec_sin',
    'sec_cos',
    'day_sin',
    'day_cos',
    'bearing_sin',
    'bearing_cos',
    'track_sin',
    'track_cos',
    'latitude_rad',
    'longitude_rad',
    'weekday_1',
    'weekday_2',
    'weekday_3',
    'weekday_4',
    'weekday_5',
    'weekday_6',
    "density_10_minutes_past",
    "density_30_minutes_past",
    "density_30_minutes_past",
]

batch_size= 298
dropout_rate = 0.5108
lr_start = 0.00016
layer_sizes = (3492, 3666)
patience_reduce = 5

if __name__ == "__main__":
    df_flights = pd.read_csv(FILE_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    arrival_times_train = df_flights.arrival_time.unique()
    df_test = pd.read_csv(FILE_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_test = seconds_till_arrival(df_test)

    train_times = np.random.choice(arrival_times_train, size=int(0.90 * len(arrival_times_train)), replace=False)
    df_train = df_flights.loc[df_flights.arrival_time.isin(train_times)]
    y_train = seconds_till_arrival(df_train)
    df_val = df_flights.loc[~df_flights.arrival_time.isin(train_times)]
    y_val = seconds_till_arrival(df_val)

    std_scaler = load(PATH_STD_SCALER)
    minmax_scaler = load(PATH_MINMAX_SCALER)

    model = VanillaNN(
        features=FEATURES,
        std_scaler=std_scaler,
        minmax_scaler=minmax_scaler,
        cols_to_scale_std = COLS_TO_SCALE_STD,
        cols_to_scale_minmax = COLS_TO_SCALE_MINMAX,
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate,
        distance_relative=True,
        lr = lr_start,
    )

    model.fit(
        df_train,
        y_train,
        df_val,
        y_val,
        batch_size=batch_size,
        patience_early=patience_reduce+3,
        patience_reduce=patience_reduce,
    )
    model.evaluate(df_test, y_test)

    model.model.save(PATH_MODEL)



