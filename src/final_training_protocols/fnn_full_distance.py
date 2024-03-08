import numpy as np
import pandas as pd
from joblib import dump, load
import os
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival
from src.processing_utils.h3_preprocessing import get_h3_index, add_density

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data_2022.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "processed", "test_data_2023_Jan-Mai.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_h3_2023_whole.bin")
PATH_MODEL = os.path.join("..", "..", "trained_models", "vanilla_nn_231016")


COLS_TO_SCALE = [
    "distance",
    "altitude",
    "geoaltitude",
    "vertical_rate",
    "groundspeed",
    "density_10_minutes_past",
    "density_30_minutes_past",
    "density_30_minutes_past",
]

FEATURES = ['distance',
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
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_flights = get_h3_index(df_flights, 4)
    df_flights = df_flights.drop(columns="geometry")
    df_flights = add_density(df_flights)
    arrival_times_train = df_flights.arrival_time.unique()
    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_test = seconds_till_arrival(df_test)
    df_test = get_h3_index(df_test, 4)
    df_test = df_test.drop(columns="geometry")
    df_test = add_density(df_test)
    train_times = np.random.choice(arrival_times_train, size=int(0.90 * len(arrival_times_train)), replace=False)
    df_train = df_flights.loc[df_flights.arrival_time.isin(train_times)]
    y_train = seconds_till_arrival(df_train)
    df_val = df_flights.loc[~df_flights.arrival_time.isin(train_times)]
    y_val = seconds_till_arrival(df_val)


    model = VanillaNN(
        features=FEATURES,
        save_scaler_file=PATH_SCALER,
        cols_to_scale=COLS_TO_SCALE,
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate,
        distance_relative=True,
    )

    model.fit(df_train, y_train, df_val, y_val, batch_size=batch_size, patience_early=patience_reduce+3, patience_reduce=patience_reduce)
    model.evaluate(df_test, y_test)

    model.model.save(PATH_MODEL)



