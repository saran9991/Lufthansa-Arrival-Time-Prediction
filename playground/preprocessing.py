import traffic
from traffic.data import opensky
from traffic.core import Traffic
from rich.pretty import pprint
import numpy as np
import pandas as pd
import copy
from datetime import datetime
pd.options.mode.chained_assignment = None
# Magic Numbers
FRANKFURT_LAT = 50.037621
FRANKFURT_LON = 8.565197
DISTANCE_AIRPORT = 4.87, # largest distance that's possible within Frankfurt airport to lat and lon
GROUNDSPEED_LANDING = 170



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

def get_complete_flights(flights, timeframe):
    """
    takes in trajectories and the timeframe in which flights must have data and returns ids of those flights which have
    start to finish in data.
    """
    # assign unique id which properly identifies a flight. A flight is uniquely identified by callsing and firstseen-
    # timestamp.
    flights_data = flights.data
    flights_data['flight_id'] = flights_data.groupby(['callsign', 'firstseen']).ngroup()
    flights_data['flight_id'] = flights_data["callsign"] + "_" + flights_data['flight_id'].astype(str)

    # create Traffic object
    traffic = Traffic(flights_data)

    # filter for onground True and ground_speed below 150
    df_flights = traffic.data[["flight_id", "groundspeed", "onground"]]
    og_flight_ids = \
        set(df_flights.loc[(flights.data.onground ==True) & (flights.data.groundspeed < GROUNDSPEED_LANDING)].flight_id)
    df_flights = df_flights.loc[df_flights.flight_id.isin(og_flight_ids)]

    # onground has to stay True. We want to replace instances of onground = True, which don't stay true.
    count_rows = df_flights[df_flights.onground==True].groupby("flight_id", as_index=False).count()
    ids = set(count_rows.loc[count_rows.onground >= 5].flight_id)
    df_flights = df_flights.loc[df_flights.flight_id.isin(ids)]

    # we remove flights which have early first timestamps on the first day in the data and move quickly already,
    # indicating that parts of the flight are on day before. These flights should be extracted when processing the day
    # before.

    # find flights that are on the first day in dataframe. Find first day and set to beginning of day. Then add 5
    # minutes
    edge_time = traffic.start_time.normalize() + pd.Timedelta(minutes=5)
    # get flight_ids of flights with times before
    early_ids = traffic.before(edge_time).flight_ids
    # find minimum groundspeed
    if len(early_ids) > 0:
        early_speed = df_flights.loc[df_flights.flight_id.isin(early_ids)].groupby("flight_id", as_index = False).min()
        early_fast_ids = early_speed.loc[early_speed.groundspeed > 200]
        ids = df_flights.loc[~df_flights.flight_id.isin(early_fast_ids)].flight_id.unique()

    # if there is a relevant timeframe, drop flights which do not have data in that timeframe
    if timeframe:
      datetime_start = pd.to_datetime(datetime.strptime(timeframe[0], '%Y-%m-%d %H:%M:%S'), utc=True)
      datetime_end = pd.to_datetime(datetime.strptime(timeframe[1], '%Y-%m-%d %H:%M:%S'), utc=True)
      ids = traffic[ids].between(datetime_start,datetime_end).flight_ids


    return traffic[ids]

def haversine(lat1, lon1,  lat2, lon2):
    """
    vectorized haversine distance. Can take in pandas series, np arrays or individual values
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km
def assign_landing_time(
        flights,
        timeframe=None,
        destination_lat=FRANKFURT_LAT,
        destination_lon=FRANKFURT_LON,
        check_distance=DISTANCE_AIRPORT, # largest distance that's possible within Frankfurt airport to lat and lon
        check_speed = GROUNDSPEED_LANDING,
):
    """
    assigns distance to airport and landing time
    """
    traffic = get_complete_flights(flights=flights, timeframe=timeframe)

    # find first onground True value of those which have speed below threshold and are sufficiently near to
    # destination coords
    df_flights = traffic.data
    df_flights["distance"] = haversine(
        df_flights["latitude"],
        df_flights["longitude"],
        destination_lat,destination_lon
    )

    df_onground = df_flights.loc[
        (df_flights.distance < check_distance) &
        (df_flights.groundspeed < check_speed) &
        (df_flights.onground == True)
        ][["flight_id","timestamp"]]

    arrival_times = (df_onground.groupby("flight_id").min())
    arrival_dict = arrival_times.to_dict()["timestamp"]
    df_flights["arrival_time"] = df_flights["flight_id"].map(arrival_dict)

    # drop all rows in which arrival time is None
    df_flights = df_flights.dropna(subset="arrival_time")

    return df_flights

def preprocess_traffic(flights, relevant_time=["1970-01-01 00:00:00", "2030-01-01 00:00:00"]):
    """
    takes in Traffic object, selects only full flights, combines flights together which belong together and assigns
    unique identifier, which uses callsign and firstseen-timestamp to uniquely identify flights. Calculates distance
    to Frankfurt airport based on the magic numbers defined at beginning of this script and assigns the exact landing
    time - time when onground switches to True, using the assign_landing time function, making sure that those are not
    false positives. Returns a data frame which ends with the last row before arrival-time.

    relevant_time can be assigned which will filter flights, which have any data-point within the specified interval.
    Interval must be a list of strings in the form "yyyy-mm-dd hh:mm:ss", "yyyy-mm-dd hh:mm:ss""
    """

    df = assign_landing_time(flights, relevant_time)
    # remove onground True values after (including) plane has landed.
    df = df.loc[df.timestamp < df.arrival_time]

    # onground is no longer needed.
    df = df.drop("onground",axis=1)

    return df
