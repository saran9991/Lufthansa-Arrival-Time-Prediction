import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.xgb_model import XGBModel
from src.processing_utils.preprocessing import seconds_till_arrival
from bayes_opt import BayesianOptimization
from src.processing_utils.preprocessing import generate_aux_columns
from src.processing_utils.h3_preprocessing import get_h3_index, add_density
import os

# Paths
PATH_TRAIN_DATA = '../../data/train_data/training_data.csv'
PATH_TEST_DATA = '../../data/test_data/test_data_2023_Jan-Mai.csv'
PATH_MODEL = os.path.join("..", "..", "trained_models", "xgb_saved_model.model")
PATH_PRE_PROCESSED_TRAIN_DATA = '../../data/pre_processed/train_data_xgb.csv'
PATH_PRE_PROCESSED_TEST_DATA = '../../data/pre_processed/test_data_xgb.csv'

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


def preprocess_xgb(data: pd.DataFrame, feature_columns) -> pd.DataFrame:
    data = data.copy()
    data = generate_aux_columns(data)
    data = get_h3_index(data, 4)
    data = add_density(data)
    data = data[feature_columns]
    return data


def tune_hyperparameters(X_train, y_train, X_test, y_test, tree_method):
    def optimize_xgb(n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree):
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        xgb_model = XGBModel(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             gamma=gamma,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             tree_method=tree_method)
        xgb_model.fit(X_train, y_train)

        mae, _ = xgb_model.evaluate(X_test, y_test)
        return -mae

    bounds = {
        'n_estimators': (100, 2000),
        'max_depth': (3, 15),
        'learning_rate': (0.01, 0.5),
        'gamma': (0, 5),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1)
    }

    optimizer = BayesianOptimization(f=optimize_xgb, pbounds=bounds, random_state=1)
    optimizer.maximize(init_points=10,
                       n_iter=100)

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_mae = -optimizer.max['target']

    print(f"Best parameters found: {best_params} with MAE: {best_mae}")

    return best_params


if __name__ == "__main__":

    print('Reading Training DataFrame...')
    X_train = pd.read_csv(PATH_TRAIN_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    y_train = seconds_till_arrival(X_train)
    X_train['flight_id'] = X_train.arrival_time
    X_train = preprocess_xgb(X_train, FEATURES)

    print('Reading Testing DataFrame...')
    X_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    y_test = seconds_till_arrival(X_test)
    X_test['flight_id'] = X_test.arrival_time
    X_test = preprocess_xgb(X_test, FEATURES)

    xgb_model = XGBModel(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, tree_method = TREE_METHOD)
    xgb_model.fit(X_train, y_train)

    mae, r2 = xgb_model.evaluate(X_test, y_test)
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, TREE_METHOD)

    #xgb_model.save_model(PATH_MODEL)
    #print(f"XGBoost Model saved to {PATH_MODEL}")
    #X_train.to_csv(PATH_PRE_PROCESSED_TRAIN_DATA)
    #X_test.to_csv(PATH_PRE_PROCESSED_TEST_DATA)

    importances = xgb_model.feature_importance_()
    df = pd.DataFrame({'features': FEATURES, 'importances': importances})
    df = df[df['features'] != 'distance']
    df = df.sort_values(by='importances', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['importances'], y=df['features'], palette="viridis")
    plt.title('Feature Importances')
    plt.show()
