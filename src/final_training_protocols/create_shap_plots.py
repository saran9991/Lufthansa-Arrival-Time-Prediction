import os
import pandas as pd
from joblib import load as load_scaler
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival

DIR_TRAINING_DATA = os.path.join("..", "..", "data", "final", "train")
FILE_TRAINING_DATA_WHOLE = os.path.join(DIR_TRAINING_DATA,"training_data_whole.csv")
FILE_TRAINING_DATA_100km = os.path.join(DIR_TRAINING_DATA,"training_data_2022_100km.csv")

DIR_TEST_DATA = os.path.join("..", "..", "data", "final", "test")
FILE_TEST_DATA_WHOLE = os.path.join(DIR_TEST_DATA,"testdata_2023_comparable.csv")
FILE_TEST_DATA_100km = os.path.join(DIR_TEST_DATA,"testdata_2023_100km_comparable.csv")

PATH_STD_SCALER_WHOLE = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_all_distances.bin")
PATH_MINMAX_SCALER_WHOLE = os.path.join("..", "..","trained_models", "scalers", "minmax_scaler_all_distances_h3.bin")

PATH_STD_SCALER_100km = os.path.join("..", "..", "trained_models", "scalers", "std_scaler_100km.bin")
PATH_MINMAX_SCALER_100km = os.path.join("..", "..","trained_models", "scalers", "minmax_scaler_100km_h3.bin")

PATH_MODEL_WHOLE =os.path.join("..", "..","trained_models", "best_models", "fnn_full_distance")
PATH_MODEL_100km =os.path.join("..", "..","trained_models", "best_models", "fnn_100km")

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
    "H3 Density 30 min",
]
if __name__ == "__main__":
    df_train_whole = pd.read_csv(FILE_TRAINING_DATA_WHOLE, parse_dates=["arrival_time", "timestamp"]).sample(n=500)
    y_train = seconds_till_arrival(df_train_whole)
    df_test_whole = pd.read_csv(FILE_TEST_DATA_WHOLE, parse_dates=["arrival_time", "timestamp"]).sample(n=500)
    y_test = seconds_till_arrival(df_test_whole)
    std_scaler = load_scaler(PATH_STD_SCALER_WHOLE)
    minmax_scaler = load_scaler(PATH_MINMAX_SCALER_WHOLE)

    fnn_full = VanillaNN(
        features = FEATURES,
        std_scaler=std_scaler,
        model_file=PATH_MODEL_WHOLE,
        minmax_scaler=minmax_scaler,
        cols_to_scale_std=COLS_TO_SCALE_STD,
        cols_to_scale_minmax=COLS_TO_SCALE_MINMAX,
        distance_relative=True
    )

    fnn_full.get_shap(df_train_whole, df_test_whole)


    df_train_100km = pd.read_csv(FILE_TRAINING_DATA_100km, parse_dates=["arrival_time", "timestamp"]).sample(n=10000)
    y_train_100km = seconds_till_arrival(df_train_100km)
    df_test_100km = pd.read_csv(FILE_TEST_DATA_100km, parse_dates=["arrival_time", "timestamp"]).sample(n=10000)
    y_test_100km = seconds_till_arrival(df_test_100km)
    std_scaler_100km = load_scaler(PATH_STD_SCALER_100km)
    minmax_scaler_100km = load_scaler(PATH_MINMAX_SCALER_100km)

    fnn_100km = VanillaNN(
        features = FEATURES,
        std_scaler=std_scaler_100km,
        model_file=PATH_MODEL_100km,
        minmax_scaler=minmax_scaler_100km,
        cols_to_scale_std=COLS_TO_SCALE_STD,
        cols_to_scale_minmax=COLS_TO_SCALE_MINMAX,
        distance_relative=True
    )

    fnn_full.get_shap(df_train_whole, df_test_whole, FEATURES= SHAP_FEATURES, file="shap_whole.png", )
    fnn_100km.get_shap(df_train_100km, df_test_100km, FEATURES= SHAP_FEATURES,file="shap_100km.png")