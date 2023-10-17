import os
import pandas as pd
from tensorflow.keras.models import load_model
from shap_fnn import get_shap

FEATURES = ['distance', 'altitude', 'geoaltitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
            'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
            'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

PATH_MODEL = os.path.join("..", "..", "..", "trained_models", "vanilla_nn_231013")
PATH_PRE_PROCESSED_TRAINING_DATA = os.path.join("..", "..", "..", "data", "pre_processed", "train_data_fcnn.csv")
PATH_PRE_PROCESSED_TEST_DATA = os.path.join("..", "..", "..", "data", "pre_processed", "test_data_fcnn.csv")


if __name__ == "__main__":
    loaded_model = load_model(PATH_MODEL)
    train_data = pd.read_csv(PATH_PRE_PROCESSED_TRAINING_DATA)
    test_data = pd.read_csv(PATH_PRE_PROCESSED_TEST_DATA)

    train_data = train_data[FEATURES]
    test_data = test_data[FEATURES]

    get_shap(train_data, test_data, loaded_model, FEATURES)
