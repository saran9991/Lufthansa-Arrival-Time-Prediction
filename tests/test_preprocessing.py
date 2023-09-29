import os
import pandas as pd
import unittest
from src.preprocessing import preprocess_traffic

class TestPreprocessing(unittest.TestCase):
    """Test all preprocessing functions """
    def test_arrival_time(self):
        """
        Tests if arrival time is properly assigned for df with multiple flights.
        Three  assertions:
        1 All flights without a proper landing time in the testdata should be dropped from df.
        2 All flights with a proper landing time should be retained in extracted df.
        2 arrival_time_true and arrival_time should be equal in all rows of extracted df.
        """
        cols_data = [
            "day",
            "firstseen",
            "hour",
            "last_position",
            "lastseen",
            "timestamp",
            "arrival_time_true"
        ]
        print(os.getcwd())
        file = "../data/testdata/test_data_arrival_time.csv"
        df = pd.read_csv(file, parse_dates= cols_data)
        df["identifier"] = df.firstseen.dt.date.astype(str) + df.callsign

        no_landing = set(df.loc[df.arrival_time_true.isna(), "identifier"].unique())
        landing = set(df.loc[df.arrival_time_true.notna(), "identifier"].unique())

        df_extract = preprocess_traffic(df.copy())
        extracted_flights = set(df_extract.identifier.unique())
        wrongly_extracted = extracted_flights.intersection(no_landing)
        wrongly_not_extracted = landing - extracted_flights

        self.assertTrue(len(wrongly_extracted) == 0)
        self.assertTrue(len(wrongly_not_extracted) == 0)
        self.assertTrue((df_extract.arrival_time == df_extract.arrival_time_true).all())

    def test_arrival_time_with_noise_remove(self):
        """
        Same as test above, just with noise-remove function
        """
        cols_data = [
            "day",
            "firstseen",
            "hour",
            "last_position",
            "lastseen",
            "timestamp",
            "arrival_time_true"
        ]
        print(os.getcwd())
        file = "../data/testdata/test_data_arrival_time.csv"
        df = pd.read_csv(file, parse_dates= cols_data)
        df["identifier"] = df.firstseen.dt.date.astype(str) + df.callsign

        no_landing = set(df.loc[df.arrival_time_true.isna(), "identifier"].unique())
        landing = set(df.loc[df.arrival_time_true.notna(), "identifier"].unique())

        df_extract = preprocess_traffic(df.copy(), remove_noise=True)
        extracted_flights = set(df_extract.identifier.unique())
        wrongly_extracted = extracted_flights.intersection(no_landing)
        wrongly_not_extracted = landing - extracted_flights

        self.assertTrue(len(wrongly_extracted) == 0)
        self.assertTrue(len(wrongly_not_extracted) == 0)
        self.assertTrue((df_extract.arrival_time == df_extract.arrival_time_true).all())




