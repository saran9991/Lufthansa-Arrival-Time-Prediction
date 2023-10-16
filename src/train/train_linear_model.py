import copy
import pandas as pd
from joblib import dump, load
import os
from src.models.linear_model import LinearModel
from src.processing_utils.preprocessing import seconds_till_arrival

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "train_data", "training_data.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "test_data", "test_data_2023_Jan-Mai.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_reg_new.bin")
PATH_MODEL =os.path.join("..", "..", "trained_models")

COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed",
                "density_10_minutes_past", "density_30_minutes_past", "density_60_minutes_past"]

FEATURES = [
    'distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday',
    'sec_sin', 'sec_cos', 'day_sin', 'day_cos', 'bearing_sin', 'bearing_cos',
    'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'density_10_minutes_past', 'density_30_minutes_past', 'density_60_minutes_past'
]

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed",
                 "density_10_minutes_past", "density_30_minutes_past", "density_60_minutes_past"]




if __name__ == "__main__":
#    scaler = load(PATH_SCALER)


    best_mae = 10000
    df_train = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_train = seconds_till_arrival(df_train)
    y_test = seconds_till_arrival(df_test)

    for i in range(1, 20):
        degrees = i
        model = LinearModel(features=FEATURES, pol_degree=degrees, cols_to_scale=COLS_TO_SCALE)
        model.fit(df_train, y_train)
        if i == 1:
            best_model = copy.deepcopy(model.model)

        mae, r2 = model.evaluate(df_test, y_test)
        print("degree", degrees, "mae", mae, r2, r2)
        if mae < best_mae:
            best_mae = mae
            best_model = copy.deepcopy(model.model)
        else:
            break

    model_file = os.path.join(PATH_MODEL, "regression_with_coord_deg_" + str(degrees - 1) + ".sav")
    dump(best_model, model_file)
