import pandas as pd
import h3
import h3pandas
from tqdm import tqdm
import time
import logging
import folium
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

def update_progress(pbar):
    pbar.update(1)
    time.sleep(0.5)

def get_h3_index(data, res):
    logging.info("Starting H3 preprocessing...")
    data = data.copy()
    with tqdm(total=5, desc="Processing", dynamic_ncols=True) as pbar:
        data.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)
        update_progress(pbar)

        data = data.h3.geo_to_h3(res)
        data['h3index'] = data.index
        update_progress(pbar)

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data[~(data.h3index == '0')]
        update_progress(pbar)

        final = data.h3.h3_to_geo_boundary()
        update_progress(pbar)

        final = final.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
        final.reset_index(inplace=True, drop=True)
        update_progress(pbar)

    logging.info('H3 Features Added')
    return final

def plot_h3(df, save_html=False, file_name="map.html"):
    """
    Create a Folium map with hexagon geometries plotted.

    :param df: DataFrame containing h3index and geometry columns
    :param save_html: Whether to save the map as an HTML file
    :param file_name: The file name of the saved HTML map if save_html is True
    :return: Folium map object
    """
    m = folium.Map(location=[df['geometry'].apply(lambda geom: geom.centroid.y).mean(),
                             df['geometry'].apply(lambda geom: geom.centroid.x).mean()],
                   zoom_start=15)

    for _, row in df.iterrows():
        folium.GeoJson(row['geometry'], name=str(row['h3index'])).add_to(m)

    if save_html:
        m.save(file_name)

    return m

def add_density(df):
    '''
    This function adds density metrics to the given DataFrame based on the number of unique aircraft
    in the same h3index within the past 10, 30, and 60 minutes.

    Parameters:
    - df: The input DataFrame containing aircraft data with timestamp, h3index, and flight_id columns.

    Returns:
    - The DataFrame with added density columns for the past 10, 30, and 60 minutes.
    '''

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    density_10_minutes_past = []
    density_30_minutes_past = []
    density_60_minutes_past = []

    h3index_flight_ids = {}

    for i in tqdm(range(len(df)), desc="Processing", unit="rows"):
        current_time = df.loc[i, 'timestamp']
        current_h3index = df.loc[i, 'h3index']
        current_flight_id = df.loc[i, 'flight_id']

        if current_h3index not in h3index_flight_ids:
            h3index_flight_ids[current_h3index] = {}

        for flight_id, timestamp in list(h3index_flight_ids[current_h3index].items()):
            if timestamp < current_time - timedelta(minutes=60):
                del h3index_flight_ids[current_h3index][flight_id]

        h3index_flight_ids[current_h3index][current_flight_id] = current_time

        count_10 = len([flight_id for flight_id, timestamp in h3index_flight_ids[current_h3index].items()
                        if flight_id != current_flight_id and timestamp >= current_time - timedelta(minutes=10)])

        count_30 = len([flight_id for flight_id, timestamp in h3index_flight_ids[current_h3index].items()
                        if flight_id != current_flight_id and timestamp >= current_time - timedelta(minutes=30)])

        count_60 = len([flight_id for flight_id in h3index_flight_ids[current_h3index]
                        if flight_id != current_flight_id])

        density_10_minutes_past.append(count_10)
        density_30_minutes_past.append(count_30)
        density_60_minutes_past.append(count_60)

    df['density_10_minutes_past'] = density_10_minutes_past
    df['density_30_minutes_past'] = density_30_minutes_past
    df['density_60_minutes_past'] = density_60_minutes_past

    return df

def weekday_column(traindata):
    weekday_df = traindata.filter(regex='^weekday_')
    traindata['weekday'] = weekday_df.idxmax(axis=1)
    traindata['weekday'] = traindata['weekday'].str.replace('weekday_', '').astype(int)
    return traindata


def calculate_average_and_merge(traindata, testdata, group_cols, col, new_col_name):
    mapping = traindata.groupby(group_cols)[col].mean()
    mapping = mapping.reset_index().rename(columns={col: new_col_name})
    return pd.merge(testdata, mapping, on=group_cols, how='left')


def test_data_h3_average(traindata, testdata):
    traindata['hour'] = traindata['timestamp'].dt.hour
    testdata['hour'] = testdata['timestamp'].dt.hour

    for group_cols, col, new_col_name in [
        (['h3index', 'hour'], 'hexbin_hourly_density', 'average_hourly_hexbin_density'),
        (['h3index', 'hour'], 'average_hourly_speed', 'average_hourly_avg_speed'),
        (['h3index', 'hour'], 'average_hourly_altitude', 'average_hourly_avg_altitude'),
    ]:
        testdata = calculate_average_and_merge(traindata, testdata, group_cols, col, new_col_name)
    testdata.rename(columns={'average_hourly_hexbin_density': 'hexbin_hourly_density', 'average_hourly_avg_speed': 'average_hourly_speed', 'average_hourly_avg_altitude':'average_hourly_altitude'}, inplace=True)
    return testdata

