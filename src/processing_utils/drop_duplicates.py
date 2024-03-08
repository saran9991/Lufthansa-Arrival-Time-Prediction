import pandas as pd
import tqdm
import os
from src.processing_utils.h3_preprocessing import get_h3_index, add_density

file = os.path.join("..", "..", "data", "processed", "training_data_2022_10sec_h3.csv")

if __name__ == "__main__":
    df = pd.read_csv(file)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(os.path.join("..", "..", "data", "processed", "training_data_2022_10sec_h3_no_dupl.csv"), index=False)
