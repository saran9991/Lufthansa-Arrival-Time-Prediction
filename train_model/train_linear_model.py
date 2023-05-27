from multiprocessing import Queue
from joblib import dump, load
from data_loader import load_data
import os
from linear_model import LinearModel


if __name__ == "__main__":
    degrees = 3
    scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg_new.bin"
    model_file = ".." + os.sep + "trained_models" + os.sep + "regression"+ " deg_" + str(degrees) +".sav"

    data_files_train = []
    for i in range(1, 13):
        month = "0" + str(i) if i < 10 else str(i)
        file = ".." + os.sep + "data" + os.sep + "Frankfurt_LH_22" + month + ".h5"
        data_files_train.append(file)
    data_files_test = [".." + os.sep + "data" + os.sep + "Frankfurt_LH_2301" + ".h5",
                       ".." + os.sep + "data" + os.sep + "Frankfurt_LH_2302" + ".h5",
                       ".." + os.sep + "data" + os.sep + "Frankfurt_LH_2303" + ".h5"]

    queue = Queue()

    load_data(queue, epochs=1, flight_files=data_files_train, threads=6,sample_fraction=0.01, random=True, quick_sample=False)
    X_train, y_train = queue.get()

    features = ['distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
                'day_cos', 'bearing_sin',
                'bearing_cos', 'track_sin', 'track_cos', 'weekday_1',
                'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

    scaler = load(scaler_file)
    model = LinearModel(cols=features, pol_degree=degrees, scaler=scaler)
    cols_to_scale = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
    X = model.preprocess(X_train, cols_to_scale)
    model.fit(X,y_train)

    load_data(queue,epochs=1,flight_files=data_files_test,threads=3,sample_fraction=0.01, random=True, quick_sample=False)
    X_test, y_test = queue.get()
    X_test = model.preprocess(X_test, cols_to_scale)

    print(model.evaluate(X_test,y_test))
    dump(model.model, model_file)