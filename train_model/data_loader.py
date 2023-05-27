from traffic.core import Traffic
from preprocessing import preprocess_traffic, seconds_till_arrival, generate_aux_columns
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
import time
import datetime
import os
from tqdm.auto import tqdm


def load_data_batch(
        file_batch,
        data_queue,
        sample_fraction=0.1,
        random = True,
        remove_noise = True,
        quick_sample = False # for testing purposes when we only want to load one day
):
    # if random false, every 1/sample_fraction row
    first_day = True
    nthrows = int(1 // sample_fraction)
    for file in file_batch:
        with h5py.File(file, 'r') as f:
            if quick_sample:
                i = 0
            for key in tqdm(list(f.keys()),desc=file):
                if quick_sample:
                    if i > 0:
                        continue
                new_flights = Traffic.from_file(file, key=key,
                                                parse_dates=["day", "firstseen", "hour", "last_position",
                                                             "lastseen", "timestamp"]).data

                if first_day:
                    df_flights = preprocess_traffic(new_flights, remove_noise=remove_noise)
                    df_flights = df_flights[
                        [
                            "distance",
                            "altitude",
                            "geoaltitude",
                            "arrival_time",
                            "timestamp",
                            "vertical_rate",
                            "groundspeed",
                            "track",
                            "latitude",
                            "longitude",
                        ]
                    ].dropna()
                    if random:
                        df_flights = df_flights.sample(frac=sample_fraction)
                    else:
                        df_flights = df_flights.iloc[::nthrows, :]
                    first_day = False
                else:
                    old_flights = pd.concat([old_flights,new_flights])
                    start = new_flights.day.min().replace(tzinfo=None)
                    end = start + datetime.timedelta(days=1)
                    relevant_time = [str(start), str(end)]
                    df_add_flights = preprocess_traffic(old_flights, relevant_time, remove_noise=remove_noise)
                    df_add_flights = df_add_flights[
                        [
                            "distance",
                            "altitude",
                            "geoaltitude",
                            "arrival_time",
                            "timestamp",
                            "vertical_rate",
                            "groundspeed",
                            "track",
                            "latitude",
                            "longitude",
                        ]
                    ].dropna()
                    del(old_flights)
                    #df_add_flights = df_add_flights.sample(frac=sample_fraction)
                    df_add_flights = df_add_flights.iloc[::nthrows, :]
                    df_flights = pd.concat([df_flights, df_add_flights])
                    del(df_add_flights)
                old_flights = new_flights
                if quick_sample:
                    i = 1


    data_queue.put(df_flights)


def load_data(queue, epochs, flight_files, threads=4, sample_fraction=0.1, random = True, remove_noise = True, quick_sample = False):
    if len(flight_files) < threads:
        print("warning fewer files than threads specified, reducing threads to number of months")
        threads = len(flight_files)

    file_batches = np.array_split(np.array(flight_files), threads)

    for i in range(epochs):
        data_queue = Queue()
        processes = []
        for batch in file_batches:
            process = Process(target=load_data_batch, args=(batch, data_queue, sample_fraction, random, remove_noise, quick_sample))
            process.start()
            processes.append(process)
        df_train = data_queue.get()
        for thread_nr in range(1, threads):
            print(thread_nr)
            df_train = pd.concat([df_train, data_queue.get()])

        df_train = generate_aux_columns(df_train)
        #shuffle the dataframe
        df_train = df_train.sample(frac=1)
        y = seconds_till_arrival(df_train)
        features = df_train.drop(columns=["arrival_time", "timestamp", "track", "latitude", "longitude"])

        while not queue.empty():
            time.sleep(2)
        queue.put((features, y))
