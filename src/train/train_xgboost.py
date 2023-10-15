import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.xgb_model import XGBModel
from src.processing_utils.preprocessing import seconds_till_arrival

# Paths
PATH_TRAIN_DATA = '../../data/train_data/training_data.csv'
PATH_TEST_DATA = '../../data/test_data/test_data_2023_Jan-Mai.csv'

FEATURES = [
    'distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday',
    'sec_sin', 'sec_cos', 'day_sin', 'day_cos', 'bearing_sin', 'bearing_cos',
    'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'density_10_minutes_past', 'density_30_minutes_past', 'density_60_minutes_past'
]

N_ESTIMATORS = 500
MAX_DEPTH = 6
TREE_METHOD ="gpu_hist"

if __name__ == "__main__":

    print('Reading Training DataFrame...')
    X_train = pd.read_csv(PATH_TRAIN_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    X_train['flight_id'] = X_train.arrival_time
    y_train = seconds_till_arrival(X_train)

    print('Reading Testing DataFrame...')
    X_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    y_test = seconds_till_arrival(X_test)

    xgb_model = XGBModel(features=FEATURES, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, tree_method = TREE_METHOD)
    xgb_model.fit(X_train, y_train)

    mae, r2 = xgb_model.evaluate(X_test, y_test, preprocess=True)
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    importances = xgb_model.feature_importance_()
    df = pd.DataFrame({'features': FEATURES, 'importances': importances})
    df = df[df['features'] != 'distance']
    df = df.sort_values(by='importances', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['importances'], y=df['features'], palette="viridis")
    plt.title('Feature Importances')
    plt.show()
