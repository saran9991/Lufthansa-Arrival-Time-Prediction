import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from src.models.xgb_model import XGBModel
from src.processing_utils.preprocessing import seconds_till_arrival
from bayes_opt import BayesianOptimization
from src.processing_utils.preprocessing import generate_aux_columns
import os


# Paths
DIR_TRAINING_DATA = os.path.join("..", "..", "data", "final", "train")
DIR_TEST_DATA = os.path.join("..", "..", "data", "final", "test")

FILE_TRAINING_DATA = os.path.join(DIR_TRAINING_DATA, "training_data_2022_100km.csv")
FILE_TEST_DATA = os.path.join(DIR_TEST_DATA,"testdata_2023_100km_comparable.csv")
PATH_MODEL = os.path.join("..", "..", "trained_models", "xgb_saved_model_100km_wide_space.model")


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
    "density_60_minutes_past",
]

N_ESTIMATORS = 500
MAX_DEPTH = 6
TREE_METHOD ="gpu_hist"
EPSILON = 1e-6

def tune_hyperparameters(X, y, tree_method):
    unique_flight_ids = X['flight_id'].unique()
    np.random.shuffle(unique_flight_ids)
    # Calculate split sizes
    total_flights = len(unique_flight_ids)
    train_size = int(0.80 * total_flights)
    train_flight_ids = unique_flight_ids[:round(train_size)]
    val_flight_ids = set(unique_flight_ids)-set(train_flight_ids)
    idx_train = X.flight_id.isin(train_flight_ids)
    idx_val = X.flight_id.isin(val_flight_ids)
    y_train = y[idx_train]
    X_train = X[idx_train]
    y_val = y[idx_val]
    X_val = X[idx_val]

    X_train = X_train[FEATURES]
    X_val = X_val[FEATURES]
    X_train.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_val.reset_index(inplace=True, drop=True)
    def optimize_xgb(n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree, reg_alpha):
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        xgb_model = XGBModel(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             gamma=gamma,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             reg_alpha=reg_alpha,
                             #tree_method=tree_method,
                             device="cuda",
                             objective="reg:absoluteerror"
                             )
        xgb_model.fit(X_train, y_train)

        # Evaluate the model on the test set
        mae, mae_relative, _ , __= xgb_model.evaluate(X_val, y_val)
        return -mae  # Objective is to minimize MAE, hence return negative MAE

    # Search space
    bounds = {
        'n_estimators': (1000, 3000),
        'max_depth': (12, 25),
        'learning_rate': (0.001, 0.02),
        'gamma': (0, 2),
        'subsample': (0.8, 1),
        'colsample_bytree': (0.8, 1),
        'reg_alpha': (10000, 20000)
    }

    # Configure the optimizer with the function and parameter bounds
    optimizer = BayesianOptimization(f=optimize_xgb, pbounds=bounds, random_state=1)
    optimizer.maximize(init_points=100, n_iter=1000)

    # Extract the best parameters and their corresponding MAE
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_mae = -optimizer.max['target']

    print(f"Best parameters found: {best_params} with MAE: {best_mae}")

    return best_params



if __name__ == "__main__":

    print('Reading Training DataFrame...')
    X_train = pd.read_csv(FILE_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    y_train = seconds_till_arrival(X_train) / (X_train['distance'] + EPSILON)
    X_train = generate_aux_columns(X_train)

    print('Reading Testing DataFrame...')
    X_test = pd.read_csv(FILE_TEST_DATA, parse_dates=["arrival_time", "timestamp"]).reset_index(drop=True)
    y_test = seconds_till_arrival(X_test) / (X_test['distance'] + EPSILON)
    X_test = generate_aux_columns(X_test)[FEATURES]

    best_params = tune_hyperparameters(X_train, y_train, TREE_METHOD) #Bayesian Optimization
    best_params_path = os.path.join('..', '..', 'trained_models', 'best_params_100km_wide_space.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f)
    print(f"Best parameters saved to {best_params_path}")

    print('Training XGB on Optimal Paramter Configuration')
    print(best_params)
    xgb_model = XGBModel(**best_params, tree_method = TREE_METHOD, device="cuda", objective="reg:absoluteerror")
    xgb_model.fit(X_train[FEATURES], y_train)
    #xgb_model = XGBModel(model_file=PATH_MODEL)
    mae, mae_relative, r2, r2_relative = xgb_model.evaluate(X_test, y_test)
    print(f"Mean Absolute Error: {mae}")
    print(f"Relative Mean Absolute Error: {mae_relative}")
    print(f"R^2 Score: {r2}")
    print(f"R^2 Score Relative: {r2_relative}")

    xgb_model.save_model(PATH_MODEL)
    print(f"XGBoost Model saved to {PATH_MODEL}")

    importances = xgb_model.feature_importance_()
    df = pd.DataFrame({'features': FEATURES, 'importances': importances})
    df = df[df['features'] != 'distance']
    df = df.sort_values(by='importances', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['importances'], y=df['features'], palette="viridis")
    plt.title('Feature Importances')
    plt.show()