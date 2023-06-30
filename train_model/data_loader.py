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
        random=True,
        remove_noise=True,
        quick_sample=False, # for testing purposes when we only want to load one day,
        keep_flight_id=False,
        distance_range=None,
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
                columns = [
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
                if keep_flight_id:
                    columns.append("flight_id")
                if first_day:
                    df_flights = preprocess_traffic(new_flights, remove_noise=remove_noise).sort_values(['flight_id', 'timestamp'])
                    df_flights = df_flights[columns].dropna()
                    if distance_range is not None:
                        df_flights = df_flights[df_flights.distance.between(distance_range[0], distance_range[1])]
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
                    df_add_flights = preprocess_traffic(old_flights, relevant_time, remove_noise=remove_noise).sort_values(['flight_id', 'timestamp'])
                    df_add_flights = df_add_flights[columns].dropna()
                    if distance_range is not None:
                        df_add_flights = df_add_flights[df_add_flights.distance.between(distance_range[0], distance_range[1])]
                    del(old_flights)
                    if random:
                        df_add_flights = df_add_flights.sample(frac=sample_fraction)
                    else:
                        df_add_flights = df_add_flights.iloc[::nthrows, :]
                    df_flights = pd.concat([df_flights, df_add_flights])
                    del(df_add_flights)
                old_flights = new_flights
                if quick_sample:
                    i = 1


    data_queue.put(df_flights)


def load_data(
        queue: Queue,
        epochs: int,
        flight_files: list,
        threads: int = 4,
        sample_fraction: float = 0.1,
        random: bool = True,
        remove_noise: bool = True,
        quick_sample: bool = False,
        save_csv: bool = False,
        csv_file: str = None,
        keep_flight_id=False,
        distance_range=None
) -> None:

    if len(flight_files) < threads:
        print("warning fewer files than threads specified, reducing threads to number of months")
        threads = len(flight_files)

    file_batches = np.array_split(np.array(flight_files), threads)

    for i in range(epochs):
        data_queue = Queue()
        processes = []
        for batch in file_batches:
            process = Process(target=load_data_batch, args=(batch, data_queue, sample_fraction, random, remove_noise, quick_sample, keep_flight_id, distance_range))
            process.start()
            processes.append(process)
        df_train = data_queue.get()
        for thread_nr in range(1, threads):
            print(thread_nr)
            df_train = pd.concat([df_train, data_queue.get()])
        if save_csv:
            if csv_file == None:
                csv_path = ".." + os.sep + "data" + os.sep + "data_" + str(time.time())[:8] + ".csv"
            else:
                csv_path = ".." + os.sep + "data" + os.sep + csv_file
            df_train.to_csv(csv_path, index=False)
        df_train = generate_aux_columns(df_train)
        #shuffle the dataframe
        df_train = df_train.sample(frac=1)
        y = seconds_till_arrival(df_train)


        while not queue.empty():
            time.sleep(2)
        queue.put((df_train, y))

def batch_generator(df: pd.DataFrame, y, batchsize, with_sample_weights=False, sample_weights=None):
    size = df.shape[0]
    while True:
        shuffled_indices = np.random.permutation(np.arange(size))
        df = df.iloc[shuffled_indices, :]
        y = y.iloc[shuffled_indices]

        if with_sample_weights:
            sample_weights = sample_weights.iloc[shuffled_indices]

        i = 0
        while i < size:
            X_batch = df.iloc[i:i+batchsize, :]
            y_batch = y.iloc[i:i+batchsize].values
            if with_sample_weights:
                sample_batch = sample_weights.iloc[i:i+batchsize].values
                yield X_batch, y_batch, sample_batch
            else:
                yield X_batch, y_batch
            i += batchsize

        X_batch = df.iloc[i:, :]
        y_batch = y.iloc[i:].values

        if with_sample_weights:
            sample_batch = sample_weights.iloc[i:].values
            yield X_batch, y_batch, sample_batch
        else:
            yield X_batch, y_batch
