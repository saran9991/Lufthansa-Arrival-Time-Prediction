from tensorflow import keras, config as tf_config
from tensorflow.keras.models import load_model
from multiprocessing import Process, Queue
from joblib import dump, load as load_scaler
from data_loader import load_data
import pandas as pd
import os

def batch_generator(df: pd.DataFrame, y, batchsize, with_sample_weights = False, sample_weights=None ):
    # we want to penalize errors more strongly if the aircraft is far away from arrival and less severely
    # when nearer
    size = df.shape[0]
    i = 0
    while i < size:
        X_batch = df.iloc[i:i+batchsize,:]
        y_batch = y.iloc[i:i+batchsize].values
        if with_sample_weights:
            sample_batch = sample_weights.iloc[i:i+batchsize].values
            yield X_batch, y_batch, sample_batch
        else:
            yield X_batch, y_batch
        i += batchsize

    X_batch = df.iloc[i:,:]
    y_batch = y.iloc[i:].values
    if with_sample_weights:
        sample_batch = sample_weights.iloc[i:i+batchsize].values
        yield X_batch, y_batch, sample_batch
    else:
        yield X_batch, y_batch


if __name__ == "__main__":
    data_files = []
    for i in range(1, 13):
        month = "0" + str(i) if i < 10 else str(i)
        file = ".." + os.sep + "data" + os.sep + "Frankfurt_LH_22" + month + ".h5"
        data_files.append(file)
    data_files_test = [".." + os.sep + "data" + os.sep + "Frankfurt_LH_2301" + ".h5",
                       ".." + os.sep + "data" + os.sep + "Frankfurt_LH_2302" + ".h5"]
    print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))
    queue = Queue()
    batch_size = 32
    scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg.bin"
    model_file = '../trained_models/model_very_thin_dropout'
    scaler = load_scaler(scaler_file)
    model = load_model(model_file)
    print(model.summary())
    load_data(queue,epochs=1,flight_files=data_files_test,threads=6,sample_fraction=0.005 )
    X_test, y_test = queue.get()
    cols_numeric = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
    X_test_numeric = X_test[cols_numeric]
    X_test[cols_numeric] = scaler.transform(X_test_numeric)
    load_data(queue, epochs=1, flight_files=data_files, threads=6, sample_fraction=0.001)
    #data_process = Process(target=load_data, args=(queue, 40, data_files, 4))
    #data_process.start()
    while True:
        X, y = queue.get()
        # weights sample importance by distance from destination
        #sample_weights = 1 / X.distance
        #weights_normalized = sample_weights / sample_weights.mean()

        # scale numeric features
        cols_numeric = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
        X_numeric = X[cols_numeric]
        X[cols_numeric] = scaler.transform(X_numeric)

        print("loading data finished. Fitting on new batch")

        while queue.empty():
            gen = batch_generator(
                df=X,
                y=y,
                batchsize=batch_size,
                #with_sample_weights=True,
                #sample_weights=weights_normalized
            )
            model.fit(gen, max_queue_size=2000, validation_data=(X_test,y_test))
            model.save(model_file)
