import tqdm
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
from sklearn.model_selection import train_test_split
from src.processing_utils.preprocessing import generate_aux_columns, seconds_till_arrival

data_path = os.path.join("../..", "data", "processed","training_data_2022_10sec_h3_no_dupl.csv")

scaler_path_std = os.path.join("../..", "trained_models", "std_scaler_100km_h3.bin")
scaler_path_minmax = os.path.join("../..", "trained_models", "minmax_scaler_100km_h3.bin")

DROP_COLUMNS = [
    "flight_id",
    "timestamp",
    "arrival_time",
    "h3index",
]
COLS_TO_SCALE_STD = [
    "distance",
    "altitude",
    "geoaltitude",
    "vertical_rate",
    "groundspeed",
]
COLS_TO_SCALE_MINMAX = [
    "density_10_minutes_past",
    "density_30_minutes_past",
    "density_60_minutes_past",
]
n_steps = 40  # number of timestamps


def add_seconds_since_last_timestep(df):
    # Calculate time difference for each group
    df['seconds_since_last_timestep'] = df.groupby('flight_id')['timestamp'].diff()

    # Convert the Timedelta to seconds (using dt.total_seconds()) and fill NaNs with 0
    df['seconds_since_last_timestep'] = df['seconds_since_last_timestep'].dt.total_seconds().fillna(0).astype(int)

    return df


def process_chunk(df, sequence_length, flight_ids_chunk, stride, stepsize, padding_value, apply_padding):
    all_data = []

    for flight_id in flight_ids_chunk:
        # Get data for the flight
        df_flight = df[df['flight_id'] == flight_id]
        flight_data = df_flight.drop(columns=DROP_COLUMNS).values

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
        results = list(tqdm.tqdm(pool.map(func, flight_id_chunks), total=len(flight_id_chunks)))

    # Flatten the list of results and concatenate all the 3D arrays together
    all_data = [item for sublist in results for item in sublist]
    three_dim_array = np.concatenate(all_data, axis=0).squeeze()
    # the second to last column tells us the seconds since last timestamp.
    # Must be zero for each first row, because there was no prior timestamp.
    three_dim_array[:, 0, -2] = 0
    return three_dim_array


if __name__ == "__main__":
    array_path_train = os.path.join("../..", "data", "processed", "timeseries_10sec_2022_100km_train.npy")
    array_path_val = os.path.join("../..", "data", "processed", "timeseries_10sec_2022_100km_val.npy")
    array_path_optim = os.path.join("../..", "data", "processed", "timeseries_10sec_2022_100km_optim.npy")

    df = pd.read_csv(data_path, parse_dates=["arrival_time", "timestamp"])
    df = df.loc[df.distance < 100]
    print(df.shape)
    arrival_days = df['arrival_time'].dt.date.unique()
    # Split the unique days of arrival into train and test sets
    arrival_days_train, arrival_days_val = train_test_split(arrival_days, test_size=0.3, random_state=42)
    # Further split the arrival_days_val into validation and test sets
    arrival_days_optim, arrival_days_val,  = train_test_split(arrival_days_val, test_size=0.3,
                                                           random_state=42)  # Assuming a 50-50 split

    flight_ids_train = df[df['arrival_time'].dt.date.isin(arrival_days_train)]['flight_id'].unique()
    flight_ids_val = df[df['arrival_time'].dt.date.isin(arrival_days_val)]['flight_id'].unique()
    flight_ids_optim = df[df['arrival_time'].dt.date.isin(arrival_days_optim)]['flight_id'].unique()

    set_train = set(flight_ids_train)
    set_val = set(flight_ids_val)
    set_optim = set(flight_ids_optim)

    # Check if intersections are empty
    is_mutually_exclusive = not (
            set_train & set_val or
            set_train & set_optim or
            set_val & set_optim
    )
    print("ids mutuallx exclusive", is_mutually_exclusive)
    print(len(set_train), len(set_val), len(set_optim), len(arrival_days))

    df = generate_aux_columns(df)
    df = add_seconds_since_last_timestep(df)
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    std_scaler.fit(df[COLS_TO_SCALE_STD])
    minmax_scaler.fit(df[COLS_TO_SCALE_MINMAX])
    dump(std_scaler, scaler_path_std)
    dump(minmax_scaler, scaler_path_minmax)
    y = seconds_till_arrival(df)
    y = y / df.distance
    df[COLS_TO_SCALE_STD] = std_scaler.transform(df[COLS_TO_SCALE_STD])
    df[COLS_TO_SCALE_MINMAX] = minmax_scaler.transform(df[COLS_TO_SCALE_MINMAX])
    df = add_seconds_since_last_timestep(df)
    df["time_to_arrival"] = y.values
    print(df.columns)
    time_series_array_val = create_time_series_array(df, n_steps, flight_ids_val, apply_padding=False, stride=20, stepsize=1)
    print(time_series_array_val.shape)
    np.save(array_path_val, time_series_array_val)
    time_series_array_optim = create_time_series_array(df, n_steps, flight_ids_optim, apply_padding=False, stride=20, stepsize=1)
    print(time_series_array_optim.shape)
    np.save(array_path_optim, time_series_array_optim)

    time_series_array_train = create_time_series_array(df, n_steps, flight_ids_train, apply_padding=False, stride=20, stepsize=1)
    print(time_series_array_train.shape)
    np.save(array_path_train, time_series_array_train)




