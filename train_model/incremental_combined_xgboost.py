import os
import pandas as pd
import xgboost as xgb
from train_model.preprocessing import assign_landing_time, generate_aux_columns, seconds_till_arrival, noise_remove
from train_model.h3_preprocessing import h3_preprocess
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class PreProcessing:

    @staticmethod
    def data_steps(data):
        data.sort_values(by=['callsign', 'timestamp'], inplace=True)
        data = noise_remove(data)
        data = assign_landing_time(data)
        data = generate_aux_columns(data)
        data['y'] = seconds_till_arrival(data)
        return data

    @staticmethod
    def preprocess_xgb(data):
        data.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)
        data = h3_preprocess(data, 6)
        X_train = data[['distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
                        'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos',
                        'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
                        "hexbin_hourly_density",
                        "average_hourly_speed", "average_hourly_altitude"]]
        y_train = data['y']
        return X_train, y_train


class XGBoost:

    def __init__(self):
        self.params, self.num_boost_round = self.get_params()

    @staticmethod
    def get_params():
        params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 8,
            'subsample': 1,
            'colsample_bytree': 1,
            'min_child_weight': 1,
            'nthread': -1
        }
        num_boost_round = 300
        return params, num_boost_round

    def train_xgb_model(self, X_train, y_train, model_name):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        print(f'Training using model: {model_name}')
        bst = xgb.train(self.params, dtrain, self.num_boost_round, xgb_model=model_name)
        return bst

    @staticmethod
    def evaluation(y_test, y_pred):
        print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))

    def incremental_xgb(self, months, test_month):
        model_name = None
        model_dir = 'trained_models/xgboost_with_h3_2022train_models'
        for month in months:
            print('\nStarting with:', month)
            file_path = os.path.join('playground/data/2022', f'{month}2022.csv')
            data = PreProcessing.data_steps(pd.read_csv(file_path).iloc[::40, :])
            print(f'Basic pre-processing completed. Adding H3 features to month: {month}')

            X_train, y_train = PreProcessing.preprocess_xgb(data)
            bst = self.train_xgb_model(X_train, y_train, model_name)

            model_name = os.path.join(model_dir, f'xgb_model_{month}.model')
            print(f'Saving model for: {month}')
            bst.save_model(model_name)

        test_file_path = os.path.join('playground/data/2023', f'{test_month}2023.csv')
        testdata = PreProcessing.data_steps(pd.read_csv(test_file_path))
        X_test, y_test = PreProcessing.preprocess_xgb(testdata)

        bst = xgb.Booster()
        bst.load_model(model_name)
        dtest = xgb.DMatrix(X_test)
        y_pred = bst.predict(dtest)

        self.evaluation(y_test, y_pred)

    def combined_xgb(self, data, test_month):
        data = PreProcessing.data_steps(data)

        X_train, y_train = PreProcessing.preprocess_xgb(data)
        dtrain = xgb.DMatrix(X_train, label=y_train)

        print('Training on combined data...')
        bst = xgb.train(self.params, dtrain, self.num_boost_round)

        test_file_path = os.path.join('playground/data/2023', f'{test_month}2023.csv')

        testdata = PreProcessing.data_steps(pd.read_csv(test_file_path))
        X_test, y_test = PreProcessing.preprocess_xgb(testdata)

        dtest = xgb.DMatrix(X_test)
        y_pred = bst.predict(dtest)

        self.evaluation(y_test, y_pred)
