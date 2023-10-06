import pandas as pd
import h3
import h3pandas
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

def update_progress(pbar):
    pbar.update(1)
    time.sleep(0.5)

def h3_preprocess(data, res):
    logging.info("Starting H3 preprocessing...")

    with tqdm(total=9, desc="Processing", dynamic_ncols=True) as pbar:
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

        final['h3_hour'] = final['timestamp'].dt.floor('H')
        grouped = final.groupby(['h3index', 'h3_hour'])
        update_progress(pbar)

        final['hexbin_hourly_density'] = grouped['flight_id'].transform('count')
        update_progress(pbar)

        final['average_hourly_speed'] = grouped['groundspeed'].transform('mean')
        update_progress(pbar)

        final['average_hourly_altitude'] = grouped['altitude'].transform('mean')
        update_progress(pbar)

        final = final.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
        final.reset_index(inplace=True, drop=True)
        update_progress(pbar)

    logging.info('H3 Features Added')
    return final

def get_h3_index(data, res):
    dfh3 = data
    dfh3.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)
    dfh3 = dfh3.h3.geo_to_h3(res)
    dfh3['h3index'] = dfh3.index  # Seperate column for h3 address
    dfh3 = dfh3.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
    dfh3.reset_index(inplace=True, drop=True)
    print('H3 Index Added')
    return dfh3

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

