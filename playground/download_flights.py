import traffic
from traffic.data import opensky
from traffic.core import Traffic
from rich.pretty import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import h5py
import os

def download_month(month: int, year: int, start_day = 1, cached=True):

    if month < 10:
        str_month = "0" + str(month)
    else:
        str_month = str(month)

    filename = "Frankfurt_LH_" + str(year)[2:] + str_month + ".h5"
    print(filename)
    next_month = month + 1 if month < 12 else  1
    str_next_month = str(next_month) if next_month > 9 else "0" + str(next_month)

    start_day_str = "0" + str(start_day) if start_day < 10 else str(start_day)
    datetime_str_start = str(year) + "-"+ str_month + "-" + start_day_str +  " 00:00:00"
    datetime_str_end = str(year) + "-" + str_next_month + "-01 00:00:00"
    datetime_current = datetime.strptime(datetime_str_start, '%Y-%m-%d %H:%M:%S')
    datetime_end = datetime.strptime(datetime_str_end, '%Y-%m-%d %H:%M:%S')

    while datetime_current < datetime_end:

        print(datetime_current)
        day = datetime_current.day
        if day < 10:
            day_str = "0" + str(day)
        else:
            day_str = str(day)

        h5_key = "LH_22" + str_month + day_str
        print(h5_key)
        str_current_day = str(datetime_current)
        str_next_day = str(datetime_current + timedelta(days = 1))
        print("getting data between "+str_current_day+ " and "+ str_next_day)

        trajectories = opensky.history(str_current_day,
                            stop= str_next_day,
                            arrival_airport ="EDDF",
                            cached=cached)
        try:
            callsigns_lh = trajectories.data.callsign.loc[trajectories.data.callsign.str.contains("DLH").astype(bool)]

            callsigns_lh = callsigns_lh.unique()
            callsigns_lh = callsigns_lh[~pd.isnull(callsigns_lh)]
            trajectories_lh = trajectories[callsigns_lh]
            trajectories_lh = trajectories_lh.assign_id().eval()
            print("add",h5_key,"to file")

            trajectories_lh.to_hdf(filename, key=h5_key, format = 'table')

        except AttributeError as e:
            print(e)

        datetime_current += timedelta(days = 1)