import os
import pandas as pd
PATH_DATA = os.path.join("..", "..", "data", "final", "train", "training_data_2022_10sec_h3_no_dupl.csv")
SAVE_PATH_CLEAN = os.path.join("..", "..", "data", "final", "train", "training_data_whole.csv")

df = pd.read_csv(PATH_DATA, parse_dates=["arrival_time", "timestamp"])
df = df.iloc[::20].reset_index(drop=True)
df.to_csv(SAVE_PATH_CLEAN, index=False)