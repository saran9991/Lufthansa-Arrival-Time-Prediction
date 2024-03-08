import os
import pandas as pd
PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data_2022_10sec_h3_no_dupl.csv")
PATH_SAVE_DATA = os.path.join("..", "..", "data", "processed", "training_data_2022_100km.csv")

if __name__ == "__main__":
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_flights = df_flights.loc[df_flights.distance < 100].sample(300000)
    df_flights.drop(columns="h3index")
    df_flights = df_flights.drop(columns="h3index")
    df_flights = df_flights.reset_index(drop=True)
    df_flights.to_csv(PATH_SAVE_DATA, index=False)