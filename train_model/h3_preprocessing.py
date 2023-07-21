import pandas as pd
import h3
import h3pandas


# You have to restart kernel from what I noticed to run it again on the dataset for some reason if such a thing happens

def h3_preprocess(data, res):
    dfh3 = data
    dfh3.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)
    dfh3 = dfh3.h3.geo_to_h3(res)
    dfh3['h3index'] = dfh3.index  # Seperate column for h3 address
    dfh3['timestamp'] = pd.to_datetime(dfh3['timestamp'])
    dfh3 = dfh3[~(dfh3.h3index == '0')]  # These are due to NaN Latitude and Longitude

    final = dfh3.h3.h3_to_geo_boundary()
    final['h3_hour'] = final['timestamp'].dt.floor('H')
    grouped = final.groupby(['h3index', 'h3_hour'])

    final['hexbin_hourly_density'] = grouped['flight_id'].transform('count')
    final['average_hourly_speed'] = grouped['groundspeed'].transform('mean')
    final['average_hourly_altitude'] = grouped['altitude'].transform('mean')
    #final['average_hourly_eta'] = grouped['y'].transform('mean')

    final = final.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
    #final.drop(columns = ['h3_hour', 'h3index', 'geometry'], inplace= True
    final.reset_index(inplace=True, drop=True)
    print('H3 Features Added')

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
    traindata = weekday_column(traindata)
    testdata  = weekday_column(testdata)

    for group_cols, col, new_col_name in [
        (['h3index', 'weekday'], 'hexbin_hourly_density', 'average_hexbin_weekday_density'),
        (['h3index', 'weekday'], 'average_hourly_speed', 'average_weekday_speed'),
        (['h3index', 'weekday'], 'average_hourly_altitude', 'average_weekday_altitude'),
        (['h3index', 'hour'], 'hexbin_hourly_density', 'average_hourly_hexbin_density'),
        (['h3index', 'hour'], 'average_hourly_speed', 'average_hourly_avg_speed'),
        (['h3index', 'hour'], 'average_hourly_altitude', 'average_hourly_avg_altitude'),
    ]:
        testdata = calculate_average_and_merge(traindata, testdata, group_cols, col, new_col_name)

    return testdata
