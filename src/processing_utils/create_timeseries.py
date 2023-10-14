import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import os
from joblib import load as load_joblib
from sklearn.model_selection import train_test_split
from src.processing_utils.preprocessing import generate_aux_columns, seconds_till_arrival

data_path = os.path.join("../..", "data", "processed","training_data_2022_10sec_sample.csv")
scaler_path = os.path.join("../..", "trained_models", "std_scaler_reg_new.bin")
COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
drop_columns = ["timestamp", "track", "latitude", "longitude", "arrival_time"]
n_steps = 40  # number of timestamps


import numpy as np
import tqdm
import multiprocessing
from functools import partial


def process_chunk(df, sequence_length, flight_ids_chunk, stride, stepsize, padding_value, apply_padding):
    all_data = []

    for flight_id in flight_ids_chunk:
        # Get data for the flight
        df_flight = df[df['flight_id'] == flight_id]
        flight_data = df_flight.drop("flight_id", axis=1).values

        if apply_padding:
            padding = np.full((sequence_length - 1, flight_data.shape[1]), padding_value)
            flight_data_padded = np.vstack((padding, flight_data))
        else:
            flight_data_padded = flight_data

        if len(flight_data_padded[::stepsize]) >= sequence_length:
            # Assuming sliding_window_view function is defined or imported in your script
            flight_array = sliding_window_view(
                flight_data_padded[::stepsize], (sequence_length, flight_data_padded.shape[1])
            )[::stride]
            all_data.append(flight_array)

    return all_data


def create_time_series_array(
        df,
        sequence_length,
        flight_ids,
        stride=1,
        stepsize=1,
        padding_value=np.nan,
        apply_padding=True,
        n_processes=6
):
    # Divide flight_ids into n_processes chunks
    flights_per_chunk = len(flight_ids) // n_processes
    flight_id_chunks = [flight_ids[i:i + flights_per_chunk] for i in range(0, len(flight_ids), flights_per_chunk)]

    # Use a Pool to parallelize the work among processes
    with multiprocessing.Pool(n_processes) as pool:
        func = partial(process_chunk, df, sequence_length, stride=stride, stepsize=stepsize,
                       padding_value=padding_value, apply_padding=apply_padding)
        results = list(tqdm.tqdm(pool.imap(func, flight_id_chunks), total=len(flight_id_chunks)))

    # Flatten the list of results and concatenate all the 3D arrays together
    all_data = [item for sublist in results for item in sublist]
    three_dim_array = np.concatenate(all_data, axis=0).squeeze()

    return three_dim_array


if __name__ == "__main__":
    array_path_train = os.path.join("../..", "data", "processed", "timeseries_20sec_2023_train_far.npy")
    array_path_val = os.path.join("../..", "data", "processed", "timeseries_20sec_2023_val_far.npy")

    df = pd.read_csv(data_path, parse_dates=["arrival_time", "timestamp"])
    arrival_days = df['arrival_time'].dt.date.unique()
    df['flight_id'] = df['flight_id'].astype(str) + '_' + df['arrival_time'].dt.date.astype(str)
    # Split the unique days of arrival into train and test sets
    arrival_days_train, arrival_days_val = train_test_split(arrival_days, test_size=0.2, random_state=42)
    arrival_days_train =arrival_days
    # Identify the flight_ids that correspond to each day of arrival
    flight_ids_train = df[df['arrival_time'].dt.date.isin(arrival_days_train)]['flight_id'].unique()
    flight_ids_val = df[df['arrival_time'].dt.date.isin(arrival_days_val)]['flight_id'].unique()

    df = generate_aux_columns(df)

    scaler = load_joblib(scaler_path)
    y = seconds_till_arrival(df)
    df = df.drop(columns=drop_columns)

    X_numeric = df[COLS_NUMERIC]
    df[COLS_NUMERIC] = scaler.transform(X_numeric)
    df["time_to_arrival"] = y.values

    #time_series_array_val = create_time_series_array(df, n_steps, flight_ids_val, apply_padding=False, stride=2, stepsize=4)
    #np.save(array_path_val, time_series_array_val)
    time_series_array_train = create_time_series_array(df, n_steps, flight_ids_train, apply_padding=False, stride=2, stepsize=4)
    np.save(array_path_train, time_series_array_train)




