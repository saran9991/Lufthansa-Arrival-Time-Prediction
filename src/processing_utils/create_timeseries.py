import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import os
import tqdm
from joblib import load as load_joblib
from sklearn.model_selection import train_test_split
from src.processing_utils.preprocessing import generate_aux_columns, seconds_till_arrival

data_path = os.path.join("../..", "data", "processed","timeseries_near_2023.csv")
scaler_path = os.path.join("../..", "trained_models", "std_scaler_reg_new.bin")
COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
drop_columns = ["timestamp", "track", "latitude", "longitude", "arrival_time"]
n_steps = 40  # number of timestamps

# Create an empty list to hold 3D arrays for each flight
import numpy as np
import tqdm

def create_time_series_array(df, sequence_length, flight_ids, padding_value=np.nan, apply_padding=True):
    all_data = []

    # Loop over each flight_id
    for flight_id in tqdm.tqdm(flight_ids):
        # Get data for the flight
        df_flight = df[df['flight_id'] == flight_id]
        flight_data = df_flight.drop("flight_id", axis=1).values

        if apply_padding:
            # Add n-1 pads at the beginning of flight_data
            padding = np.full((sequence_length-1, flight_data.shape[1]), padding_value)
            flight_data_padded = np.vstack((padding, flight_data))
        else:
            flight_data_padded = flight_data

        # Ensure enough data is present for at least one window
        if len(flight_data_padded) >= sequence_length:
            # Create a 3D array for this flight using strides
            # Assuming sliding_window_view function is defined or imported in your script
            flight_array = sliding_window_view(flight_data_padded, (sequence_length, flight_data_padded.shape[1]))

            # Append to the list
            all_data.append(flight_array)

    # Concatenate all the 3D arrays together
    three_dim_array = np.concatenate(all_data, axis=0).squeeze()
    return three_dim_array

if __name__ == "__main__":
    array_path_train = os.path.join("../..", "data", "processed", "timeseries_20secpad_2023_train.npy")
    array_path_val = os.path.join("../..", "data", "processed", "timeseries_20secpad_2023_val.npy")

    df = pd.read_csv(data_path, parse_dates=["arrival_time", "timestamp"])
    arrival_days = df['arrival_time'].dt.date.unique()
    df['flight_id'] = df['flight_id'].astype(str) + '_' + df['arrival_time'].dt.date.astype(str)
    # Split the unique days of arrival into train and test sets
    arrival_days_train, arrival_days_val = train_test_split(arrival_days, test_size=0.2, random_state=42)

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

    time_series_array_val = create_time_series_array(df, n_steps, flight_ids_val)
    np.save(array_path_val, time_series_array_val)
    time_series_array_train = create_time_series_array(df, n_steps, flight_ids_train)
    np.save(array_path_train, time_series_array_train)




