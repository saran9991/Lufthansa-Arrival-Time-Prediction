from traffic.core import Traffic
from preprocessing import preprocess_traffic,  generate_aux_columns
import h5py
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from multiprocessing import Process, Queue
import time


def create_arrival_df(
        file_batch,
        queue
):

    first_file = True
    for file in file_batch:
        first_day = True
        with h5py.File(file, 'r') as f:

            for key in tqdm(list(f.keys()),desc=file):
                if first_day:

                    flights = Traffic.from_file(file, key=key,
                                                    parse_dates=["day", "firstseen", "hour", "last_position",
                                                                 "lastseen", "timestamp"]).data
                    flights = preprocess_traffic(flights)
                    flights = flights.loc[flights.timestamp == flights.arrival_time]
                    first_day = False
                else:
                    new_flights = Traffic.from_file(file, key=key,
                                                    parse_dates=["day", "firstseen", "hour", "last_position",
                                                                 "lastseen", "timestamp"]).data
                    new_flights = preprocess_traffic(new_flights)
                    new_flights = new_flights.loc[new_flights.timestamp == new_flights.arrival_time]
                    flights = pd.concat([flights, new_flights])


        df_arrival_new = generate_aux_columns(flights)

        columns = [
            'callsign',
            'day',
            'firstseen',
            'groundspeed',
            'hour',
            'icao24',
            'last_position',
            'lastseen',
            'latitude',
            'longitude',
            'origin',
            'track',
            'flight_id',
            'arrival_time',
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
            'weekday_0',
            'weekday_1',
            'weekday_2',
            'weekday_3',
            'weekday_4',
            'weekday_5',
            'weekday_6'
        ]
        df_arrival_new = df_arrival_new[columns]
        if first_file:
            df_total = df_arrival_new
            first_file=False
        else:
            df_total = pd.concat([df_total, df_arrival_new])
    while not queue.empty():
        time.sleep(2)
    queue.put(df_total)
    print("added df")


def load_data(
        flight_files: list,
        threads: int = 4,
) -> None:

    if len(flight_files) < threads:
        print("warning fewer files than threads specified, reducing threads to number of months")
        threads = len(flight_files)

    file_batches = np.array_split(np.array(flight_files), threads)


    data_queue = Queue()
    processes = []
    for batch in file_batches:
        process = Process(target=create_arrival_df, args=(batch, data_queue))
        process.start()
        processes.append(process)
    df = data_queue.get()
    for thread_nr in range(1, threads):
        print(thread_nr)
        df = pd.concat([df, data_queue.get()])
    df = df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    path_data = os.path.join("..", "..", "data", "raw")
    files = os.listdir(path_data)
    file_batch = [os.path.join(path_data, file) for file in files]
    df = load_data(file_batch, 6)
    path_arrivals = os.path.join("..", "..", "data", "processed", "arrivals_2022.csv")
    df.to_csv(path_arrivals, index=False)