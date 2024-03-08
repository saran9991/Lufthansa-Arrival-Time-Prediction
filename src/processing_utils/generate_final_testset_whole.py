from joblib import load
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.processing_utils.preprocessing import generate_aux_columns, seconds_till_arrival
PATH_DATA = os.path.join("..", "..", "data", "processed")
PATH_TEST_DATA = os.path.join(PATH_DATA, "data_2023_10sec_h3.csv")
scaler_path_std = os.path.join("..", "..", "trained_models", "std_scaler_all_distances_h3.bin")
scaler_path_minmax = os.path.join("..", "..", "trained_models", "minmax_scaler_all_distances_h3.bin")
PATH_TO_SAVE_DF = os.path.join(PATH_DATA, "testdata_2023_comparable.csv")
PATH_TO_SAVE_NP = os.path.join(PATH_DATA, "testdata_2023_comparable.npy")


std_scaler = load(scaler_path_std)
minmax_scaler = load(scaler_path_minmax)

DROP_COLUMNS = [
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

def add_seconds_since_last_timestep(df):
    # Calculate time difference for each group
    df['seconds_since_last_timestep'] = df.groupby('flight_id')['timestamp'].diff()

    # Convert the Timedelta to seconds (using dt.total_seconds()) and fill NaNs with 0
    df['seconds_since_last_timestep'] = df['seconds_since_last_timestep'].dt.total_seconds().fillna(0).astype(int)

    return df

# Function to filter rows within each flight group
def filter_valid_rows(group):
    # Get rows that have at least 40 prior data points in the same flight
    return group.iloc[39:]

# First for within 100km
df = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
df = df.sort_values(by= ["flight_id", "timestamp"]).reset_index(drop=True)
df = df.iloc[::3].reset_index(drop=True)
# Filter rows by flight_id without resetting the index
valid_rows_by_flight = df.groupby('flight_id').apply(filter_valid_rows).droplevel(0)
sampled_rows = valid_rows_by_flight.sample(n=100000)  # Adjust sample size as needed
sampled_df = sampled_rows.copy().drop(columns="h3index").reset_index(drop=True)

sampled_df.to_csv(PATH_TO_SAVE_DF, index=False)

y = seconds_till_arrival(df)
y = y / df.distance
df = generate_aux_columns(df)
df[COLS_TO_SCALE_STD] = std_scaler.transform(df[COLS_TO_SCALE_STD])
df[COLS_TO_SCALE_MINMAX] = minmax_scaler.transform(df[COLS_TO_SCALE_MINMAX])
df = add_seconds_since_last_timestep(df)
df["time_to_arrival"] = y.values
df = df.drop(columns=DROP_COLUMNS)

# Function to create timeseries data for a sampled row
def create_timeseries(row):
    flight = row['flight_id']
    idx = row.name  # Get the index of the current row in the original DataFrame

    # Extract the 40 prior rows within the same flight_id from the original dataframe
    return df.loc[(df['flight_id'] == flight) &
                  (df.index <= idx) &
                  (df.index > idx - 40)].drop(columns="flight_id").values


# Function to process a chunk of data
def process_chunk(chunk, total, display_pbar=False):
    if display_pbar:
        pbar = tqdm(total=total, position=0)
    else:
        pbar = None  # No progress bar for other chunks

    def wrapper(row):
        if pbar:
            pbar.update(1)  # update the progress bar
        return create_timeseries(row)

    result = chunk.apply(wrapper, axis=1).tolist()
    if pbar:
        pbar.close()  # close the progress bar once done

    return result


# Splitting the dataframe into chunks
n_chunks = 6
chunks = np.array_split(sampled_rows, n_chunks)

# Disable the default joblib progress bar to avoid interference with tqdm
with Parallel(n_jobs=n_chunks, verbose=0) as parallel:
    # Use display_pbar=True only for the first chunk
    results = parallel(
        delayed(process_chunk)(chunk, len(chunk), display_pbar=(i == 0)) for i, chunk in enumerate(chunks))

# Combine the results and convert to numpy array
timeseries_data = np.concatenate(results, axis=0)

# Check the shape of the 3D array
print(timeseries_data.shape)  # Should be (10, 40, number_of_columns) in this example
timeseries_data[:, 0, -2] = 0
# Create a separate DataFrame for the sampled rows

np.save(PATH_TO_SAVE_NP, timeseries_data)
