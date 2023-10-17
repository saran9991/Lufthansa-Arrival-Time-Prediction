from traffic.core import Traffic
from preprocessing import preprocess_traffic, seconds_till_arrival, generate_aux_columns
from h3_preprocessing import get_h3_index, add_density as calculate_h3_density
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
        random=False,
        remove_noise=True,
        quick_sample=False, # for testing purposes when we only want to load one day,
        distance_range=None,
        ids=None,
        keep_cols=None,
        h3_density=False,
        h3_res=4,
):
    # if random false, every 1/sample_fraction row
    first_day = True
    nthrows = int(1 // sample_fraction)

    # for the first flight in the batch, we need the previous day also, which is not in the batch
    # check if the current file is the very last in the dir
    directory = os.path.dirname(file_batch[0])
    files_in_dir = os.listdir(directory)
    df_created = False
    if os.path.basename(file_batch[0]) != files_in_dir[0]:
        index_current = files_in_dir.index(os.path.basename(file_batch[0]))
        file = os.path.join(directory,files_in_dir[index_current-1])
        with h5py.File(file, 'r') as f:
            keys = list(f.keys())
            key = keys[-1]

            new_flights = Traffic.from_file(file, key=key,
                                            parse_dates=["day", "firstseen", "hour", "last_position",
                                                         "lastseen", "timestamp"]).data
            if new_flights.shape[0] > 0:
                new_flights["flight_id"] = new_flights["callsign"] + "_" + new_flights['firstseen'].astype(str)

            if ids:
                new_flights = new_flights.loc[new_flights.flight_id.isin(ids)]
            if new_flights.shape[0] > 0:
                first_day = False
                old_flights = new_flights

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
                if new_flights.shape[0] < 1:
                    continue

                new_flights["flight_id"] = new_flights["callsign"] + "_" + new_flights['firstseen'].astype(str)

                if ids:
                    new_flights = new_flights.loc[new_flights.flight_id.isin(ids)]
                    if new_flights.shape[0] < 1:
                        continue
                if first_day:
                    old_flights = new_flights.copy()
                    df_flights = preprocess_traffic(new_flights, remove_noise=remove_noise).sort_values(['flight_id', 'timestamp'])
                    if keep_cols:
                        df_flights=df_flights[keep_cols]
                    df_flights = df_flights.dropna()
                    if distance_range is not None:
                        df_flights = df_flights[df_flights.distance.between(distance_range[0], distance_range[1])]
                    if random:
                        df_flights = df_flights.sample(frac=sample_fraction)
                    else:
                        df_flights = df_flights.iloc[::nthrows, :]
                    first_day = False

                else:
                    old_flights = pd.concat([old_flights,new_flights])
                    min_date = new_flights.day.min()

                    if pd.isna(min_date):
                        print("Warning: Minimum date is NaT!")
                    start = new_flights.day.min().replace(tzinfo=None)
                    end = start + datetime.timedelta(days=1)
                    relevant_time = [str(start), str(end)]
                    df_add_flights = preprocess_traffic(old_flights, relevant_time, remove_noise=remove_noise).sort_values(['flight_id', 'timestamp'])
                    if keep_cols:
                        df_add_flights=df_add_flights[keep_cols]
                    df_add_flights = df_add_flights.dropna()

                    if distance_range is not None:
                        df_add_flights = df_add_flights[df_add_flights.distance.between(distance_range[0], distance_range[1])]
                    del(old_flights)
                    if random:
                        df_add_flights = df_add_flights.sample(frac=sample_fraction)
                    else:
                        df_add_flights = df_add_flights.iloc[::nthrows, :]
                        filename = key +".csv"
                        save_file = os.path.join("..", "..", "data", "processed", "10seconds", filename)
                        df_add_flights.to_csv(save_file, index=False)
                    if df_created:
                        df_flights = pd.concat([df_flights, df_add_flights])
                        del (df_add_flights)
                    else:
                        df_flights = df_add_flights
                        df_created = True
                    old_flights = new_flights
                if quick_sample:
                    i = 1
    while not data_queue.empty():
        time.sleep(2)
    data_queue.put(df_flights)


def load_data(
        flight_files: list,
        threads: int = 4,
        sample_fraction: float = 0.1,
        random: bool = False,
        remove_noise: bool = True,
        quick_sample: bool = False,
        distance_range=None,
        ids=None,
        keep_cols=None,
        h3_density=False,
        h3_res=4,
) -> None:

    if len(flight_files) < threads:
        print("warning fewer files than threads specified, reducing threads to number of months")
        threads = len(flight_files)

    file_batches = np.array_split(np.array(flight_files), threads)


    data_queue = Queue()
    processes = []
    for batch in file_batches:
        process = Process(
            target=load_data_batch,
            args=(
                batch,
                data_queue,
                sample_fraction,
                random,
                remove_noise,
                quick_sample,
                distance_range,
                ids,
                keep_cols,
            )
        )
        process.start()
        processes.append(process)
    df_train = data_queue.get()
    for thread_nr in range(1, threads):
        print(thread_nr)
        df_train = pd.concat([df_train, data_queue.get()])
    df_train = df_train.reset_index(drop=True)
    return df_train

if __name__ == "__main__":
    columns = [
        "flight_id",
        "timestamp",
        "distance",
        "altitude",
        "geoaltitude",
        "arrival_time",
        "vertical_rate",
        "groundspeed",
        "track",
        "latitude",
        "longitude",
    ]
    FILENAME = "training_data_2022_10sec.csv"
    queue = Queue()
    dirname = os.path.join("..", "..", "data", "raw")
    save_file = os.path.join("..", "..", "data", "processed", FILENAME)
    files = [os.path.join(dirname,file) for file in os.listdir(dirname)][7:8]
    df = load_data(files, threads=1, keep_cols=columns, sample_fraction=0.1)
    #df.to_csv(save_file, index= False)

