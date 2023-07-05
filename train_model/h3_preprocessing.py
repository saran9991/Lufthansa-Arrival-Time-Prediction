import pandas as pd
import h3
import h3pandas


# You have to restart kernel from what I noticed to run it again on the dataset for some reason if such a thing happens

def h3_preprocess(data, res):
    dfh3 = data
    dfh3.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)

    print('Doing geo_to_h3')
    dfh3 = dfh3.h3.geo_to_h3(res)

    dfh3['h3index'] = dfh3.index  # Seperate column for h3 address
    dfh3['timestamp'] = pd.to_datetime(dfh3['timestamp'])
    dfh3 = dfh3[~(dfh3.h3index == '0')]  # These are due to NaN Latitude and Longitude

    print('Doing h3_to_geo_boundary')
    final = dfh3.h3.h3_to_geo_boundary()

    final['h3_hour'] = final['timestamp'].dt.floor('H')
    print('dt.hour completed')

    grouped = final.groupby(['h3index', 'h3_hour'])
    print('Grouping according to hour and h3index completed')

    final['hexbin_hourly_density'] = grouped['flight_id'].transform('count')
    final['average_hourly_speed'] = grouped['groundspeed'].transform('mean')
    final['average_hourly_altitude'] = grouped['altitude'].transform('mean')
    final['average_hourly_eta'] = grouped['y'].transform('mean')

    final.rename(columns={'lat': 'latitude', 'lng': 'longitude'}, inplace=True)
    return final
