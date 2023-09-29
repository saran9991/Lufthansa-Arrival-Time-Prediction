import copy
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from src.models.linear_model import LinearModel
from src.processing_utils.preprocessing import seconds_till_arrival

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "processed", "training_data.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_reg_new.bin")
PATH_MODEL =os.path.join("..", "..", "trained_models")

COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

FEATURES = ['distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
            'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
            'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]




if __name__ == "__main__":
    scaler = load(PATH_SCALER)


    best_mae = 1000
    df_train = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_train = seconds_till_arrival(df_train)
    y_test = seconds_till_arrival(df_test)
    model_file_old = os.path.join(PATH_MODEL, "regression_with_coord_deg_6.sav")
    model_old = LinearModel(cols=FEATURES, pol_degree=6, model_file=model_file_old, scaler=scaler, cols_to_scale=COLS_TO_SCALE)
    print(np.absolute((model_old.predict(df_train) -y_train)).mean())
    for i in range(1, 20):
        degrees = i
        model = LinearModel(cols=FEATURES, pol_degree=degrees, scaler=scaler, cols_to_scale=COLS_TO_SCALE)
        X_train_processed = model.preprocess(df_train, COLS_TO_SCALE)

        model.fit(X_train_processed, y_train)
        if i == 1:
            best_model = copy.deepcopy(model.model)

        X_test_processed = model.preprocess(df_test, COLS_TO_SCALE)

        mae, r2 = model.evaluate(X_test_processed, y_test)
        print("degree", degrees, "mae", mae, r2, r2)
        if mae < best_mae:
            best_mae = mae
            best_model = copy.deepcopy(model.model)
        else:
            break

    model_file = os.path.join(PATH_MODEL, "regression_with_coord_deg_" + str(degrees - 1) + ".sav")
    dump(best_model, model_file)
