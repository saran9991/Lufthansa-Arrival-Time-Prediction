import pandas as pd
import holidays
import datetime

def is_holiday(row, holiday_dates):
    return row['timestamp'].date() in holiday_dates

def add_holiday(df, year):
    # Get Frankfurt holidays for the specified year
    hessen_holidays = holidays.Germany(years=year, state='HE')
    hessen_holidays_dates = [date for date in hessen_holidays.keys()]

    # Add the holiday column to the DataFrame
    df['holiday'] = df.apply(is_holiday, axis=1, args=(hessen_holidays_dates,))
    return df

def add_month_weekday(df):
    df['month'] = df['timestamp'].apply(lambda x: x.month)
    df['weekday'] = df['timestamp'].apply(lambda x: x.weekday()) # Monday: 0, Sunday: 6
    return df
