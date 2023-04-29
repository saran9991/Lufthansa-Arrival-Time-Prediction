from traffic.core import Traffic
import numpy as np
import pandas as pd
import holidays
from datetime import datetime

pd.options.mode.chained_assignment = None
# Magic Numbers
FRANKFURT_LAT = 50.037621
FRANKFURT_LON = 8.565197
DISTANCE_AIRPORT = 4.87,  # largest distance that's possible within Frankfurt airport to lat and lon
GROUNDSPEED_LANDING = 170


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
        set(df_flights.loc[
                (flights.data.onground == True) & (flights.data.groundspeed < GROUNDSPEED_LANDING)].flight_id)
    df_flights = df_flights.loc[df_flights.flight_id.isin(og_flight_ids)]

    # onground has to stay True. We want to replace instances of onground = True, which don't stay true.
    count_rows = df_flights[df_flights.onground == True].groupby("flight_id", as_index=False).count()
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
        early_speed = df_flights.loc[df_flights.flight_id.isin(early_ids)].groupby("flight_id", as_index=False).min()
        early_fast_ids = early_speed.loc[early_speed.groundspeed > 200]
        ids = df_flights.loc[~df_flights.flight_id.isin(early_fast_ids)].flight_id.unique()

    # if there is a relevant timeframe, drop flights which do not have data in that timeframe
    if timeframe:
        datetime_start = pd.to_datetime(datetime.strptime(timeframe[0], '%Y-%m-%d %H:%M:%S'), utc=True)
        datetime_end = pd.to_datetime(datetime.strptime(timeframe[1], '%Y-%m-%d %H:%M:%S'), utc=True)
        ids = traffic[ids].between(datetime_start, datetime_end).flight_ids

    return traffic[ids]


def haversine(lat1, lon1, lat2, lon2):
    """
    vectorized haversine distance. Can take in pandas series, np arrays or individual values
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    km = 6367 * dist  # 6367 for distance in KM for miles use 3958
    return km


def assign_landing_time(
        flights,
        timeframe=None,
        destination_lat=FRANKFURT_LAT,
        destination_lon=FRANKFURT_LON,
        check_distance=DISTANCE_AIRPORT,  # largest distance that's possible within Frankfurt airport to lat and lon
        check_speed=GROUNDSPEED_LANDING,
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
        destination_lat, destination_lon
    )

    df_onground = df_flights.loc[
        (df_flights.distance < check_distance) &
        (df_flights.groundspeed < check_speed) &
        (df_flights.onground == True)
        ][["flight_id", "timestamp"]]

    arrival_times = (df_onground.groupby("flight_id").min())
    arrival_dict = arrival_times.to_dict()["timestamp"]
    df_flights["arrival_time"] = df_flights["flight_id"].map(arrival_dict)

    # drop all rows in which arrival time is None
    df_flights = df_flights.dropna(subset="arrival_time")

    return df_flights


def add_month_weekday(df):
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.isocalendar().day - 1  # Monday: 0, Sunday: 6
    return df


def generate_holidays(timestamp, years):
    # Get Frankfurt holidays for the specified year
    hessen_holidays = holidays.Germany(years=years, state='HE')
    hessen_holidays_dates = np.array([date for date in hessen_holidays.keys()])

    # Add the holiday column to the DataFrame
    holiday = timestamp.dt.date.isin(hessen_holidays_dates)
    return holiday


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
    df = df.drop("onground", axis=1)

    return df


def generate_dummy_columns(df: pd.DataFrame):
    weekday_dummies = pd.get_dummies(df.weekday, prefix='weekday')
    for i in range(7):
        weekday_name = "weekday_" + str(i)
        if weekday_name not in weekday_dummies.columns:
            weekday_dummies[weekday_name] = 0
    weekday_dummies = weekday_dummies.sort_index(axis=1)
    df[list(weekday_dummies.columns)] = weekday_dummies


    month_dummies = pd.get_dummies(df.month, prefix='month')
    for i in range(1, 13):
        month_name = "month_" + str(i)
        if month_name not in df.columns:
            df[month_name] = 0
    month_dummies = month_dummies.sort_index(axis=1)
    df[list(month_dummies.columns)] = month_dummies

    df["holiday"] = df["holiday"].astype(int)
    return df


def generate_cyclical_second(timestamp):
    total_seconds = 86400
    seconds_scaled = (3600 * timestamp.dt.hour + 60 * timestamp.dt.minute + timestamp.dt.second) / total_seconds
    sec_sin = np.sin(2 * np.pi * seconds_scaled)
    sec_cos = np.cos(2 * np.pi * seconds_scaled)
    return sec_sin, sec_cos


def generate_cyclical_day(timestamp):
    days_scaled = timestamp.dt.dayofyear / 365
    day_sin = np.sin(2 * np.pi * days_scaled)
    day_cos = np.cos(2 * np.pi * days_scaled)
    return day_sin, day_cos

def generate_aux_columns(df):
    df["weekday"] = df["timestamp"].dt.isocalendar().day - 1  # Monday: 0, Sunday: 6
    df["month"] = df["timestamp"].dt.month
    years = list(df.timestamp.dt.year.unique())
    df["holiday"] = generate_holidays(df.timestamp, years)
    df["sec_sin"], df["sec_cos"] = generate_cyclical_second(df.timestamp)
    df["day_sin"], df["day_cos"] = generate_cyclical_day(df.timestamp)
    df = generate_dummy_columns(df)
    df = df.drop(columns=["weekday", "month"])
    return df

def seconds_till_arrival(flights_data: pd.DataFrame):
    time_till_arrival = flights_data["arrival_time"] - flights_data["timestamp"]
    seconds = time_till_arrival.dt.total_seconds()
    return seconds
