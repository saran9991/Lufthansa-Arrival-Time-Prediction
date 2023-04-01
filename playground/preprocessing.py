import traffic
from traffic.data import opensky
from traffic.core import Traffic
from rich.pretty import pprint
import pandas as pd
import copy
from datetime import datetime

def get_edge_flights(flights):
    """
    gets the flight id's of flights with data from the first 5 minutes of the day and last 5 minutes of the day.
    """
    time_str_early = "00:05:00"
    time_str_late = "23:55:00"
    time_early = datetime.strptime(time_str_early, '%H:%M:%S').time()
    time_late = datetime.strptime(time_str_late, '%H:%M:%S').time()
    early_flight_ids = flights.data.loc[flights.data.timestamp.dt.time<=time_early].flight_id.unique()
    late_flight_ids = flights.data.loc[(flights.data.timestamp.dt.time>=time_late)].flight_id.unique()    
    normal_flight_ids = set(flights.flight_ids) - set(early_flight_ids) - set(late_flight_ids)
    
    return normal_flight_ids, early_flight_ids, late_flight_ids

def get_complete_flights(flights):
    """
    takes in trajectories and returns ids of those flights which have start to finish in data.
    """
    
    # filter for onground True and ground_speed below 150
    df_flights = flights.data[["flight_id", "groundspeed", "onground"]]
    og_flight_ids = set(df_flights.loc[(flights.data.onground ==True) & (flights.data.groundspeed < 150)].flight_id)
    df_flights = df_flights.loc[df_flights.flight_id.isin(og_flight_ids)]
    
    # onground has to stay True. We want to replace instances of onground = True, which don't stay true.
    count_rows = df_flights[df_flights.onground==True].groupby("flight_id", as_index=False).count()
    ids = set(count_rows.loc[count_rows.onground >= 5].flight_id)    
    df_flights = df_flights.loc[df_flights.flight_id.isin(ids)]
    
    # we remove flights which have early first timestamps and move quickly already, indicating that parts of the flight are on day before
    _, early, __ = get_edge_flights(flights)
    print(early)
    # find minimum groundspeed
    if len(early) > 0:
        print(len(early))
        early_speed = df_flights.loc[df_flights.flight_id.isin(early)].groupby("flight_id", as_index = False).min()
        print(early_speed)
        early_fast_ids = early_speed.loc[early_speed.groundspeed > 200]
        ids = df_flights.loc[~df_flights.flight_id.isin(early_fast_ids)].flight_id.unique()

    return ids


