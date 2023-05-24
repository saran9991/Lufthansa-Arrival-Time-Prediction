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

FRANKFURT_LAT = 50.037621
FRANKFURT_LON = 8.565197
DISTANCE_AIRPORT = 4.87,  # largest distance that's possible within Frankfurt airport to lat and lon
GROUNDSPEED_LANDING = 170


def get_complete_flights(df, timeframe):
    """
    takes in trajectories dataframe and the timeframe in which flights must have data and returns ids of those flights
    which have start to finish in data.
    """
    # assign unique id which properly identifies a flight. A flight is uniquely identified by callsing and firstseen-
    # timestamp.
    df['flight_id'] = df.groupby(['callsign', 'firstseen']).ngroup()
    df['flight_id'] = df["callsign"] + "_" + df['flight_id'].astype(str)

    # filter for onground True and ground_speed below 150
    df_flights = df[["flight_id", "groundspeed", "onground"]]
    og_flight_ids = \
        set(df_flights.loc[
                (df.onground == True) & (df.groundspeed < GROUNDSPEED_LANDING)].flight_id)
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
    edge_time = df.day.min() + pd.Timedelta(minutes=5)
    # get flight_ids of flights with times before
    df.loc[df.timestamp < edge_time].flight_id.unique()
    early_ids = df.loc[df.timestamp < edge_time].flight_id.unique()
    # find minimum groundspeed
    if len(early_ids) > 0:
        early_speed = df_flights.loc[df_flights.flight_id.isin(early_ids)].groupby("flight_id", as_index=False).min()
        early_fast_ids = early_speed.loc[early_speed.groundspeed > 200]
        ids = df_flights.loc[~df_flights.flight_id.isin(early_fast_ids)].flight_id.unique()

    # if there is a relevant timeframe, drop flights which do not have data in that timeframe
    if timeframe:
        datetime_start = pd.to_datetime(datetime.strptime(timeframe[0], '%Y-%m-%d %H:%M:%S'), utc=True)
        datetime_end = pd.to_datetime(datetime.strptime(timeframe[1], '%Y-%m-%d %H:%M:%S'), utc=True)
        ids = df.loc[(df.timestamp.between(datetime_start, datetime_end)) & (df.flight_id.isin(ids))].flight_id.unique()

    return df.loc[df.flight_id.isin(ids)]


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
        df,
        timeframe=None,
        destination_lat=FRANKFURT_LAT,
        destination_lon=FRANKFURT_LON,
        check_distance=DISTANCE_AIRPORT,  # largest distance that's possible within Frankfurt airport to lat and lon
        check_speed=GROUNDSPEED_LANDING,
):
    """
    assigns distance to airport and landing time
    """
    df = get_complete_flights(df, timeframe=timeframe)

    # find first onground True value of those which have speed below threshold and are sufficiently near to
    # destination coords
    df["distance"] = haversine(
        df["latitude"],
        df["longitude"],
        destination_lat, destination_lon
    )

    df_onground = df.loc[
        (df.distance < check_distance) &
        (df.groundspeed < check_speed) &
        (df.onground == True)
        ][["flight_id", "timestamp"]]

    arrival_times = (df_onground.groupby("flight_id").min())
    arrival_dict = arrival_times.to_dict()["timestamp"]
    df["arrival_time"] = df["flight_id"].map(arrival_dict)

    # drop all rows in which arrival time is None. Reassign correct datatype. In Large datasets timestamps can
    # get messed up
    df = df.dropna(subset="arrival_time").astype(
        {"timestamp": "datetime64[ns, UTC]", "arrival_time": "datetime64[ns, UTC]"}
    )

    return df


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


def preprocess_traffic(df_flights, relevant_time=["1970-01-01 00:00:00", "2030-01-01 00:00:00"]):
    """
    takes in Traffic object, selects only full flights, combines flights together which belong together and assigns
    unique identifier, which uses callsign and firstseen-timestamp to uniquely identify flights. Calculates distance
    to Frankfurt airport based on the magic numbers defined at beginning of this script and assigns the exact landing
    time - time when onground switches to True, using the assign_landing time function, making sure that those are not
    false positives. Returns a data frame which ends with the last row before arrival-time.

    relevant_time can be assigned which will filter flights, which have any data-point within the specified interval.
    Interval must be a list of strings in the form "yyyy-mm-dd hh:mm:ss", "yyyy-mm-dd hh:mm:ss""
    """

    df_flights = assign_landing_time(df_flights, relevant_time)
    # remove onground True values after (including) plane has landed.
    df_flights = df_flights.loc[df_flights.timestamp < df_flights.arrival_time]

    # onground is no longer needed.
    df_flights = df_flights.drop("onground", axis=1)

    return df_flights


def generate_dummy_columns(df: pd.DataFrame, with_month=True):
    weekday_dummies = pd.get_dummies(df.weekday, prefix='weekday')
    for i in range(7):
        weekday_name = "weekday_" + str(i)
        if weekday_name not in weekday_dummies.columns:
            weekday_dummies[weekday_name] = 0
    weekday_dummies = weekday_dummies.sort_index(axis=1)
    df[list(weekday_dummies.columns)] = weekday_dummies

    if with_month:
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


def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate the longitude difference
    delta_lon = lon2_rad - lon1_rad

    # Calculate the bearing
    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)
    bearing_rad = np.arctan2(y, x)

    # Convert the bearing to degrees
    bearing_deg = np.degrees(bearing_rad)

    # Normalize the bearing to the range of 0 to 360 degrees
    bearing_deg = (bearing_deg + 360) % 360
    return bearing_deg


def generate_aux_columns(df, with_month=False):
    df["weekday"] = df["timestamp"].dt.isocalendar().day - 1  # Monday: 0, Sunday: 6
    if with_month:
        df["month"] = df["timestamp"].dt.month

    years = list(df.timestamp.dt.year.unique())
    df["holiday"] = generate_holidays(df.timestamp, years)
    df["sec_sin"], df["sec_cos"] = generate_cyclical_second(df.timestamp)
    df["day_sin"], df["day_cos"] = generate_cyclical_day(df.timestamp)
    bearing = calculate_bearing(FRANKFURT_LAT, FRANKFURT_LON, df.latitude, df.longitude)
    df["bearing_sin"] = np.sin(bearing * 2 * np.pi / 360)
    df["bearing_cos"] = np.cos(bearing * 2 * np.pi / 360)
    df = generate_dummy_columns(df, with_month=with_month)

    df = df.drop(columns=["weekday", "month"]) if with_month else df.drop(columns=["weekday"])
    df.reset_index(drop=True, inplace=True)

    return df


def seconds_till_arrival(flights_data: pd.DataFrame):
    time_till_arrival = flights_data["arrival_time"] - flights_data["timestamp"]
    seconds = time_till_arrival.dt.total_seconds()
    return seconds


def noise_remove(data):
    data_shifted = data.shift(1)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data_shifted['timestamp'] = pd.to_datetime(data_shifted['timestamp'])

    data['timestamp'] = data['timestamp'].fillna(pd.NaT)
    data_shifted['timestamp'] = data_shifted['timestamp'].fillna(pd.NaT)

    data['time_difference'] = (data['timestamp'] - data_shifted['timestamp']).dt.total_seconds()
    data['altitude_difference'] = data['altitude'] - data_shifted['altitude']
    data['geoaltitude_difference'] = data['geoaltitude'] - data_shifted['geoaltitude']

    data['onground_prev'] = data['onground'].shift(1)
    data['onground_next'] = data['onground'].shift(-1)
    data['time_difference_prev'] = data['time_difference'].shift(1)
    data['time_difference_next'] = (data['timestamp'].shift(-1) - data['timestamp']).dt.total_seconds()

    # Conditions based on sampling of data every 5 seconds. Change time difference condition for higher sampling rate
    cond1 = (data['altitude'] > 45000) | (data['geoaltitude'] > 45000)
    cond2 = (data['altitude_difference'].abs() > 2000) & (data['time_difference'] <= 12)
    cond3 = (data['geoaltitude_difference'].abs() > 2000) & (data['time_difference'] <= 12)
    cond4 = (data['altitude_difference'].abs() > 5000) & (data['time_difference'] <= 30)
    cond5 = (data['geoaltitude_difference'].abs() > 5000) & (data['time_difference'] <= 30)
    cond6 = (data['onground'] == True) & (data['groundspeed'] > 200) & (data['altitude'] > 10000)
    cond7 = (data['onground_prev'] != data['onground']) & (data['onground_next'] != data['onground']) & (
            data['time_difference_prev'] <= 15) & (data['time_difference_next'] <= 15)

    # cond6 = (data['altitude'].isna()) & (data['onground'] == True)

    drop_conditions = cond1 | cond2 | cond3 | cond4 | cond5 | cond6 | cond7
    data = data[~drop_conditions]
    data.drop(
        columns=['time_difference', 'onground_prev', 'onground_next', 'time_difference_prev', 'time_difference_next'],
        inplace=True)  # 'altitude_difference', 'geoaltitude_difference' could be useful later

    return data

