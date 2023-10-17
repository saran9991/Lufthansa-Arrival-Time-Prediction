import os
from shap_xgboost import get_shap
import xgboost as xgb
import pandas as pd
from joblib import load

FEATURES = [
    'distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday',
    'sec_sin', 'sec_cos', 'day_sin', 'day_cos', 'bearing_sin', 'bearing_cos',
    'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'density_10_minutes_past', 'density_30_minutes_past', 'density_60_minutes_past'
]

PATH_MODEL = os.path.join("..", "..", "..", "trained_models", "xgb_saved_model.model")
PATH_PRE_PROCESSED_TRAINING_DATA = os.path.join("..", "..", "..", "data", "pre_processed", "train_data_xgb.csv")
PATH_PRE_PROCESSED_TEST_DATA = os.path.join("..", "..", "..", "data", "pre_processed", "test_data_xgb.csv")


if __name__ == "__main__":
    model = load(PATH_MODEL)
    df_train = pd.read_csv(PATH_PRE_PROCESSED_TRAINING_DATA)
    df_test = pd.read_csv(PATH_PRE_PROCESSED_TEST_DATA)

    get_shap(df_train, df_test, model, FEATURES)
