import pandas as pd
import tqdm
import os
from src.processing_utils.h3_preprocessing import get_h3_index, add_density

dirname = os.path.join("..", "..", "data", "processed", "10seconds")

if __name__ == "__main__":
   files =os.listdir(dirname)
   dfs = []
   i = 0
   for file in tqdm.tqdm(files):
       i += 1
       df = pd.read_csv(os.path.join(dirname,file),  parse_dates=["arrival_time", "timestamp"])
       dfs.append(df)

   df_concat = pd.concat(dfs, ignore_index=True)
   df_concat = df_concat.drop_duplicates().reset_index(drop=True)
   df_concat.to_csv(os.path.join(dirname, "training_data_2023_10sec_low_dense_removed.csv"))
   df_concat = get_h3_index(df_concat, 4)
   df_concat = add_density(df_concat)
   df_concat.to_csv(os.path.join(dirname, "training_data_2023_10sec_h3.csv"), index=False)