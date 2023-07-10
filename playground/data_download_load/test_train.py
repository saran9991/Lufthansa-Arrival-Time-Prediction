from traffic.core import Traffic
from preprocessing import get_complete_flights, preprocess_traffic
import h5py
import numpy as np
import pandas as pd
from tensorflow import keras, config as tf_config
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from multiprocessing import Process, Queue
from joblib import dump, load as load_scaler
import time



def seconds_till_arrival(flights_data: pd.DataFrame):
    time_till_arrival = flights_data["arrival_time"]- flights_data["timestamp"]
    seconds = time_till_arrival.dt.total_seconds()
    return seconds

def build_sequential(lr, input_dims, output_dims, layerSizes, activation, loss):
    model = keras.Sequential()
    model.add (keras.layers.Input(shape=input_dims))
    for size in layerSizes:
        model.add(keras.layers.Dense(size))
        model.add(keras.layers.LeakyReLU(alpha=0.05))

    model.add(keras.layers.Dense(output_dims,activation=activation))

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)

    return model

def load_data_batch(month_batch, data_queue):
    first = True
    for month in month_batch:
        file = "Frankfurt_LH_22" + month + ".h5"
        for day in range(1, 32):

            if (month in ["06", "09", "11"] and day == 31) or (month == "02" and day == 29):
                continue

            str_day = str(day) if day > 9 else "0" + str(day)
            key = "LH_22" + month + str_day
            with h5py.File(file, 'r') as f:
                print(key)
                if key in list(f.keys()):
                    flights = Traffic.from_file(file, key=key,
                                                parse_dates=["day", "firstseen", "hour", "last_position",
                                                             "lastseen", "timestamp"])
                else:
                    continue
                try:
                    df_flights = preprocess_traffic(flights)
                except:
                    continue
                df = df_flights[
                    ["distance", "altitude", "geoaltitude", "arrival_time", "timestamp", "vertical_rate",
                     "groundspeed"]].dropna()
                df_sample = df.sample(frac=0.1)
                if not first:
                    df_train = pd.concat([df_sample, df_train])
                else:
                    df_train = df_sample
                    first = False
    data_queue.put(df_train)


def load_data(queue, epochs, months, threads = 4):
    if len(months)<threads:
        print("warning fewer months than threads specified, reducing threads to number of months")
        threads = len(months)
    month_batches = np.array_split(np.array(months), threads)

    for i in range(epochs):
        data_queue = Queue()
        processes = []
        for batch in month_batches:
            process = Process(target=load_data_batch, args=(batch, data_queue))
            process.start()
            processes.append(process)
        df_train = data_queue.get()
        for thread_nr in range(1,threads):
            print(thread_nr)
            df_train = pd.concat([df_train, data_queue.get()])
        X = np.array(df_train[["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]])  # we only take the feature
        y = seconds_till_arrival(df_train)

        while not queue.empty():
            time.sleep(2)
        queue.put((X, y))


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))
    queue = Queue()
    batch_size = 128
    file = "model_large"

    scaler = load_scaler("new_scaler.bin")
    #model = build_sequential(lr=0.0001, input_dims=(5,),output_dims=1, layerSizes=(1024,512, 256),activation=None,loss="MSE")
    model = load_model("model_large")
    months = ["01", "02", "03", "05", "06", "07", "08", "09", "10", "11"]
    load_data(queue, epochs=1, months=months, threads=5)
    data_process = Process(target=load_data, args=(queue,10, months, 3))
    data_process.start()
    while True:
       X, y = queue.get()
       X_scaled = scaler.transform(X)

       while queue.empty():
           model.fit(X_scaled, y, batch_size=batch_size)
           model.save(file)








