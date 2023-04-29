from tensorflow import keras, config as tf_config
from tensorflow.keras.models import load_model
from multiprocessing import Process, Queue
from joblib import dump, load as load_scaler
from data_loader import load_data
import pandas as pd
import os

def batch_generator(df: pd.DataFrame, y, batchsize):
    size = df.shape[0]
    i = 0
    while i < size:
        yield df.iloc[i:i+batchsize,:], y.iloc[i:i+batchsize].values
        i += batchsize
    yield df.iloc[i:,:], y.iloc[i:].values

if __name__ == "__main__":
    data_files = []
    for i in range(1, 13):
        month = "0" + str(i) if i < 10 else str(i)
        file = ".." + os.sep + "data" + os.sep + "Frankfurt_LH_22" + month + ".h5"
        data_files.append(file)
    print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))
    queue = Queue()
    batch_size = 128
    scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg.bin"
    model_file = ".." + os.sep + "trained_models" + os.sep + "model_with_cat"
    scaler = load_scaler(scaler_file)
    model = load_model(model_file)
    print(model.summary())
    load_data(queue, epochs=1, flight_files=data_files, threads=6)
    data_process = Process(target=load_data, args=(queue, 40, data_files, 4))
    data_process.start()
    while True:
        X, y = queue.get()

        print("loading data finished. Fitting on new batch")

        while queue.empty():
            gen = batch_generator(X, y, batch_size)
            model.fit(gen, max_queue_size=2000)
            model.save(model_file)
