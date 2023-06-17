import copy
from multiprocessing import Queue
from joblib import dump, load
from data_loader import load_data
import os
from fit_model import prepare_data
from linear_model import LinearModel


if __name__ == "__main__":

    best_mae = 1000
    X_train, y_train, _ = prepare_data()
    X_test, y_test, __ = prepare_data(test=True)
    for i in range(1, 20):
        degrees = i


        scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg_new.bin"
        features = ['distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
                    'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
                    'weekday_1','weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

        scaler = load(scaler_file)
        model = LinearModel(cols=features, pol_degree=degrees, scaler=scaler)
        cols_to_scale = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
        X = model.preprocess(X_train, cols_to_scale)
        model.fit(X,y_train)
        if i == 1:
            best_model = copy.deepcopy(model.model)

        X_test_processed = model.preprocess(X_test, cols_to_scale)

        mae, r2 = model.evaluate(X_test_processed,y_test)
        print("degree", degrees, "mae", mae, r2, r2)
        if mae < best_mae:
            best_mae = mae
            best_model = copy.deepcopy(model.model)
        else:
            break

    model_file = ".." + os.sep + "trained_models" + os.sep + "regression_with_coord_deg_" + str(degrees-1) + ".sav"
    dump(best_model, model_file)