from traffic.core import Traffic
from src.processing_utils.preprocessing import preprocess_traffic, seconds_till_arrival, generate_aux_columns
from src.processing_utils.h3_preprocessing import get_h3_index, add_density
import h5py
import numpy as np
import pandas as pd
import os


dirname = os.path.join("..", "..", "data", "raw")
file = os.path.join(dirname,"Frankfurt_LH_2205.h5")
with h5py.File(file, 'r') as f:
    keys = list(f.keys())
    key = keys[-1]

    new_flights = Traffic.from_file(file, key=key,
                                    parse_dates=["day", "firstseen", "hour", "last_position",
                                                 "lastseen", "timestamp"]).data
new_flights["flight_id"] = new_flights["callsign"] + "_" + new_flights['firstseen'].astype(str)
df_dense = get_h3_index(new_flights, 4)
df_dense = add_density(df_dense)
df_thin = new_flights.iloc[::200,:]
df_thin = get_h3_index(df_thin, 4)
df_dense_shared = df_dense.iloc[df_dense.index.isin(df_thin.index)]
df_compare = df_dense_shared.iloc[:, -3:].compare(df_dense_shared.iloc[:, -3:])
df_diff = df_thin.iloc[:, -3:] - df_dense_shared.iloc[:, -3:]
print("here")