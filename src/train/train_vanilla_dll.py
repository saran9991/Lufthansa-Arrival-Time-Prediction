import numpy as np
import pandas as pd
from joblib import dump, load
import os
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data_0617.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "processed", "final_testset_vanilla.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_reg_new.bin")
PATH_MODEL = os.path.join("..", "..", "trained_models", "vanilla_nn_231013")

COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

FEATURES = ['distance', 'altitude', 'geoaltitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
            'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
            'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

layer_sizes = (3050, 774, 512)
dropout = 0.07672833230417865

if __name__ == "__main__":
    scaler = load(PATH_SCALER)
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    arrival_times_train = df_flights.arrival_time.unique()
    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_test = seconds_till_arrival(df_test)
    train_times = np.random.choice(arrival_times_train, size=int(0.90 * len(arrival_times_train)), replace=False)
    df_train = df_flights.loc[df_flights.arrival_time.isin(train_times)]
    y_train = seconds_till_arrival(df_train)
    df_val = df_flights.loc[~df_flights.arrival_time.isin(train_times)]
    y_val = seconds_till_arrival(df_val)


    model = VanillaNN(
        features=FEATURES,
        scaler=scaler,
        cols_to_scale=COLS_TO_SCALE,
        layer_sizes=layer_sizes,
        dropout_rate=dropout,
        distance_relative=True,
    )

    model.fit(df_train, y_train, df_val, y_val, batch_size=256, patience_early=5, patience_reduce=3)
    model.evaluate(df_test, y_test)

    model.model.save(PATH_MODEL)



