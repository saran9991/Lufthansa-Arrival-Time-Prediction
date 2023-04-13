import pandas as pd
from geopy.distance import great_circle


def assign_origin(complete_flight_dataset, airports):
    """
    Assigns the origin airport to each flight in the given dataset based on the nearest airport.

    :param complete_flight_dataset: pandas.DataFrame containing complete flight data
    :param airports: pandas.DataFrame containing airport data
    :return: pandas.DataFrame containing complete flight data with an additional 'origin' column
    """

    # First rows indicates the first entry of each aircraft in the complete_aircraft dataframe.
    # If the flight is complete indeed in the complete_aircraft dataframe, the first row should indicate that it's near the the source airport
    # Adding another column to first rows - origin
    # Origin column indicates the origin airport for the particular aircraft
    # Finally this is merged with complete flight dataset where now datasets are merged on flight ID and origin column is applied to all rows of the same aircraft
    first_rows = get_first_rows(complete_flight_dataset)
    first_rows['origin'] = first_rows.apply(lambda row: find_nearest_airport(row['latitude'], row['longitude'], airports), axis=1)
    complete_flight_dataset = pd.merge(complete_flight_dataset, first_rows[['flight_id', 'origin']], on='flight_id', how='left')

    # Dropping and renaming unnecessary column names after dataset merging
    complete_flight_dataset.drop('origin_y', axis=1, inplace=True)
    complete_flight_dataset = complete_flight_dataset.rename(columns={'origin_x': 'origin'})

    return complete_flight_dataset


def get_first_rows(complete_flight_dataset):
    """
    Returns the first rows of each flight in the given dataset, sorted by timestamp.

    :param complete_flight_dataset: pandas.DataFrame containing complete flight data
    :return: pandas.DataFrame containing the first rows of each flight
    """
    return complete_flight_dataset.sort_values(by='timestamp').groupby('flight_id').first().reset_index()


def find_nearest_airport(latitude, longitude, airports):
    """
    Finds the nearest airport to the given coordinates.

    :param latitude: float representing the latitude of the coordinates
    :param longitude: float representing the longitude of the coordinates
    :param airports: pandas.DataFrame containing airport data
    :return: string representing the nearest airport's ident code
    """

    # Minimum distance is first assigned to infinity
    # All rows in the airports dataset containing coordinates of all airports of the world are iterated through
    # The coordinates of each airport are compared with the aircraft's firstrow coordinates ( taking off coordinates )
    # This comparison is done with respect to distance between the airport and the aircraft's coordinates
    # The closest airport assigned as aircraft's origin
    min_distance = float('inf')
    nearest_airport = None

    for _, row in airports.iterrows():
        airport_coords = (row['latitude_deg'], row['longitude_deg'])
        aircraft_coords = (latitude, longitude)
        distance = great_circle(airport_coords, aircraft_coords).km

        if distance < min_distance:
            min_distance = distance
            nearest_airport = row['ident']

    return nearest_airport
