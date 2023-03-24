import traffic
from traffic.data import opensky
from traffic.core import Traffic
from rich.pretty import pprint
import pandas as pd
import copy
from datetime import datetime

def get_edge_flights(flights):
    time_str_early = "00:05:00"
    time_str_late = "23:55:00"
    time_early = datetime.strptime(time_str_early, '%H:%M:%S').time()
    time_late = datetime.strptime(time_str_late, '%H:%M:%S').time()
    early_flight_ids = flights.data.loc[flights.data.timestamp.dt.time<=time_early].flight_id.unique()
    late_flight_ids = flights.data.loc[(flights.data.timestamp.dt.time>=time_late)].flight_id.unique()    
    normal_flight_ids = set(flights.flight_ids) - set(early_flight_ids) - set(late_flight_ids)
    
    return normal_flight_ids, early_flight_ids, late_flight_ids

def get_combined_ids(flights, flights_before, flights_after, early_flight_ids, late_flight_ids):
    #check if early flights are late flights in other df.
    # create id that can be comparable
    if len(early_flight_ids) != 0:
        _, __, before_late = get_edge_flights(flights_before)    
    
        if len(before_late) != 0:
            id_flights_early = flights[early_flight_ids].data.icao24 + flights[early_flight_ids].data.callsign
            id_flights_before_late = flights_before[before_late].data.icao24 + flights_before[before_late].data.callsign

            identifiers_early = set(id_flights_early) & set(id_flights_before_late)
        else:
            identifier_early = {}
    else:
        identifier_early = {}
    
    #check if late flights are early flights in other df.
    # create id that can be comparable
    if len(late_flight_ids) != 0:
        _, after_early,__  = get_edge_flights(flights_after)   
    
        if len(after_early) != 0:
            id_flights_late = flights[late_flight_ids].data.icao24 + flights[late_flight_ids].data.callsign
            id_flights_after_early = flights_after[after_early].data.icao24 + flights_after[after_early].data.callsign

            identifiers_late = set(id_flights_late) & set(id_flights_after_early)
        else:
            identifiers_late = {}
    else:
        identifiers_late = {}
           
    
    return identifier_early, identifiers_late
    
def get_full_flights(flights, flights_before = None, flights_after=None):
    # A full flight goes from start till onground switches to True. Many flights don't even have onground variables with True value
    # get all flights which have timestamps 2 seconds before end of day and 2 seconds after start of day. Those must be combined with 
    # other data
    normal_flight_ids, early_flight_ids, late_flight_ids = get_edge_flights(flights)
    combined_id_early, combined_id_late = get_combined_ids(flights, flights_before, flights_after, early_flight_ids, late_flight_ids)
    
    if len(combined_id_early) != 0:
        pass
    if len(combined_id_late) != 0:
        pass
    
    flights_total = flights[normal_flight_ids]
    # get only flights which have onground values of true
    flight_ids_grouped = flights_total.data.groupby("flight_id", as_index=False).mean(numeric_only=True)
    flight_ids_onground = flight_ids_grouped.loc[flight_ids_grouped.onground > 0].flight_id.values
    return flight_ids_onground

def get_arrival_time(df: pd.DataFrame):
    # only get flights, which have both values for True and False.
    flights_in_air = df[["callsign", "icao24", "timestamp", "onground"]].groupby(["callsign", "icao24"], as_index=False).mean("onground")
    flights_in_air = flights_in_air.loc[flights_in_air.onground < 1]
    flights_in_air = flights_in_air.callsign + flights_in_air.icao24    
    arrival_times = df[df.onground == True][["callsign", "icao24", "timestamp"]].groupby(["callsign", "icao24"], as_index=False).min()    
    arrival_times = arrival_times.loc[(arrival_times.callsign + arrival_times.icao24).isin(flights_in_air)]
    data = copy.copy(df)
    data["id"] = data["callsign"]+data["icao24"]
    arrival_times["id"] = arrival_times["callsign"]+arrival_times["icao24"]
    arrivals = arrival_times[["id","timestamp"]].set_index("id").squeeze()
    
    return df["id"].map(arrivals)