import unittest
from preprocessing import get_edge_flights
import traffic
from traffic.core import Traffic
from rich.pretty import pprint
import pandas as pd
import numpy as np
from datetime import datetime

file = "Frankfurt_LH_2301.h5"
trajectories = [
    Traffic.from_file(file, key = "LH_230115", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230116", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230117", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230122", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230123", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230124", parse_dates=["day","firstseen", "hour", "last_position","lastseen","timestamp"]),
]

class TestPreprocessing(unittest.TestCase):

    
    def test_edge_flights(self):
        """
        edge flights are those that have values at the end of the day or at the beginning of the day. This tests if all edge flights are captured 
        and non edge flights are not captured
        """
        
        time_str_early = "00:00:02"
        time_str_late = "23:59:58"
        time_early = datetime.strptime(time_str_early, '%H:%M:%S').time()
        time_late = datetime.strptime(time_str_late, '%H:%M:%S').time()
        

        for trajectory in trajectories:
            early_flights_true = set(trajectory.data.loc[trajectory.data["timestamp"].dt.time <= time_early].flight_id.unique())
            late_flights_true = set(trajectory.data.loc[trajectory.data["timestamp"].dt.time >= time_late].flight_id.unique())
            
            _, early_flights_assert, late_flights_assert = get_edge_flights(trajectory)
            print(early_flights_true,"\n",early_flights_assert)
            print(late_flights_true,"\n",late_flights_assert)
                     
            self.assertEqual(early_flights_true, set(early_flights_assert))
            self.assertEqual(late_flights_true, set(late_flights_assert))
            
    def test_get_combined_ids(self):
        """
        for the edge flights, we need identifiers to find them in the other dataframes. The identifiers are concatenated icao24 and callsign.
        This test uses mock examples and real examples
        """
        
        
        
        
        
