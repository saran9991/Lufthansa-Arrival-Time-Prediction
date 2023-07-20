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
    Traffic.from_file(file, key = "LH_230115", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230116", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230117", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230122", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230123", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
    Traffic.from_file(file, key = "LH_230124", parse_dates=[
        "day","firstseen", "hour", "last_position","lastseen","timestamp"]),
]

class TestPreprocessing(unittest.TestCase):
    
    def test_edge_flights(self):
        """
        edge flights are those that have values at the end of the day or at the beginning of the day. This tests if
        all edge flights are captured and non edge flights are not captured
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
        for the edge flights, we need identifiers to find them in the other dataframes. The identifiers are
        concatenated icao24 and callsign.
        """
        flights_middle = trajectories[2]
        flights_before = trajectories[1]
        flights_after = trajectories[3]
        m_normal, m_early, m_late = get_edge_flights(flights_middle)
        b_normal, b_early, b_late = get_edge_flights(flights_before)
        a_normal, a_early, a_late = get_edge_flights(flights_after)

        # create identifieres
        ids_m_early = (flights_middle[m_early].data.icao24 + flights_middle[m_early].data.callsign).unique()
        ids_b_late = (flights_before[b_late].data.icao24 + flights_before[b_late].data.callsign).unique()


        
        
        
        
        
