import pandas as pd
import os
from src.processing_utils.preprocessing import preprocess_traffic, generate_aux_columns
from src.processing_utils.h3_preprocessing import get_h3_index, add_density
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import pytz
from geopy.distance import geodesic
import folium
import numpy as np


PATH_DATA = os.path.join("..", "..", "data", "2022", "combined_data.csv")

FEATURES = [
    'distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday',
    'sec_sin', 'sec_cos', 'day_sin', 'day_cos', 'bearing_sin', 'bearing_cos',
    'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'density_10_minutes_past', 'density_30_minutes_past', 'density_60_minutes_past'
]

def preprocess_plot_data(data, h3_processing = False):
    df = data.copy()
    df = preprocess_traffic(df)
    df = generate_aux_columns(df)
    if(h3_processing):
        df = get_h3_index(df, 4)
        df = add_density(df)
    return df


def plot_origin_density(data):
    unique_flights_df = data.drop_duplicates(subset='flight_id')
    origin_counts = unique_flights_df['origin'].value_counts().head(20)  # Considering top 20 busiest origins

    colors = mcolors.LinearSegmentedColormap.from_list("custom", ["#1F77B4", "#9467BD", "#FF7F0E"], N=20)
    bar_colors = colors(np.linspace(0, 1, 20))

    plt.figure(figsize=(12, 8))
    origin_counts.plot(kind='bar', color=bar_colors)

    plt.title('Top 20 Busiest Origin Airports for Lufthansa Flights in 2022')
    plt.xlabel('Origin Airport')
    plt.ylabel('Number of Flights')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(r'C:\Users\saran\Desktop\Plots\busiest_origins.png')
    plt.show()


def plot_flight_times_average(data):
    df = data[data['groundspeed'] >= 170]

    df_grouped = df.groupby('flight_id').agg({'timestamp': ['min', 'max'], 'origin': 'first'}).reset_index()
    df_grouped.columns = ['flight_id', 'min_timestamp', 'max_timestamp', 'origin']
    df_grouped['travel_time'] = (df_grouped['max_timestamp'] - df_grouped['min_timestamp']).dt.total_seconds() / 60

    avg_travel_time = df_grouped['travel_time'].mean()
    median_travel_time = df_grouped['travel_time'].median()

    plt.figure(figsize=(12, 8))
    sns.histplot(df_grouped['travel_time'], bins=50, kde=True, color='salmon', edgecolor='k', linewidth=0.5)

    top_flights = df_grouped.nlargest(5, 'travel_time')
    for index, row in top_flights.iterrows():
        plt.annotate(row['origin'], (row['travel_time'], 0), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.title('Travel Times of Lufthansa Flights Landing in Frankfurt (2022)', fontsize=20, pad=20)
    plt.xlabel('Travel Time (minutes)', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(x=avg_travel_time, color='darkcyan', linestyle='--')
    plt.text(avg_travel_time + 5, max(plt.ylim()) * 0.9, f'Average: {avg_travel_time:.2f} mins', color='darkcyan',
             fontsize=12)
    plt.axvline(x=median_travel_time, color='#DAA520', linestyle='--')
    plt.text(median_travel_time + 5, max(plt.ylim()) * 0.8, f'Median: {median_travel_time:.2f} mins', color='#DAA520',
             fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(r'C:\Users\saran\Desktop\Plots\annotated_travel_times.png')

    plt.show()


def plot_busy_hours(data):
    data['timestamp'] = data['timestamp'].dt.tz_convert('Europe/Berlin')

    data['hour'] = data['timestamp'].dt.hour

    data['weekday'] = np.dot(
        data[['weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']].values,
        np.array([0, 1, 2, 3, 4, 5, 6]))

    flight_density = data.groupby(['weekday', 'hour', 'flight_id']).size().groupby(['weekday', 'hour']).count().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(flight_density, cmap="magma")
    plt.title('Flight Density Over Days and Hours')
    plt.savefig(r'C:\Users\saran\Desktop\Plots\average_flights_heatmap.png')
    plt.show()


def plot_gs_alt_100_FRA(data):
    data = data.dropna()
    FRA_COORDINATES = (50.110924, 8.682127)

    def calculate_distance(row):
        return geodesic(FRA_COORDINATES, (row['latitude'], row['longitude'])).km

    data['distance_FRA'] = data.apply(calculate_distance, axis=1)
    filtered_df = data[data['distance_FRA'] <= 100]

    bins = np.arange(0, 101, 10)
    filtered_df['distance_bin'] = np.digitize(filtered_df['distance_FRA'], bins)
    avg_data = filtered_df.groupby('distance_bin')[['altitude', 'groundspeed']].mean()

    plt.figure(figsize=(20, 6))
    sns.set_style("whitegrid")

    ax1 = plt.gca()

    sns.lineplot(data=avg_data, x=bins[:-1], y='altitude', label='Average Altitude',
                 color='darkcyan', marker='o', markersize=8, ax=ax1)

    ax1.set_title('Average Altitude and Groundspeed vs Distance to Frankfurt Airport', fontsize=18, pad=20)
    ax1.set_xlabel('Distance to Frankfurt Airport (km)', fontsize=15, labelpad=15)
    ax1.set_ylabel('Average Altitude (feet)', fontsize=15, labelpad=15, color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()

    sns.lineplot(data=avg_data, x=bins[:-1], y='groundspeed', label='Average Groundspeed',
                 color='orange', marker='o', markersize=8, ax=ax2)

    ax2.set_ylabel('Average Groundspeed (knots)', fontsize=15, labelpad=15, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    altitude_range = avg_data['altitude'].max() - avg_data['altitude'].min()
    groundspeed_range = avg_data['groundspeed'].max() - avg_data['groundspeed'].min()

    ax1.set_ylim(avg_data['altitude'].min() - 0.1 * altitude_range, avg_data['altitude'].max() + 0.1 * altitude_range)
    ax2.set_ylim(avg_data['groundspeed'].min() - 0.1 * groundspeed_range,
                 avg_data['groundspeed'].max() + 0.1 * groundspeed_range)

    for i in range(len(bins) - 1):
        ax1.text(bins[i], avg_data['altitude'].iloc[i], f"{avg_data['altitude'].iloc[i]:.0f}",
                 color='darkblue', fontsize=9, ha='center', va='bottom')
        ax2.text(bins[i], avg_data['groundspeed'].iloc[i], f"{avg_data['groundspeed'].iloc[i]:.0f}",
                 color='darkorange', fontsize=9, ha='center', va='bottom')

    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    plt.savefig(r'C:\Users\saran\Desktop\Plots\avg_altitude_groundspeed_non_normalized_dual_axis.png',
                bbox_inches='tight')

    plt.show()


def plot_heatmap_near_FRA(data, feature='altitude'):
    data = data.dropna()
    FRA_COORDINATES = (50.110924, 8.682127)

    def calculate_distance(row):
        return geodesic(FRA_COORDINATES, (row['latitude'], row['longitude'])).km

    data['distance_FRA'] = data.apply(calculate_distance, axis=1)

    filtered_df = data[data['distance_FRA'] <= 200]
    heatmap_data, xedges, yedges = np.histogram2d(filtered_df['longitude'], filtered_df['latitude'],
                                                  weights=filtered_df[feature],
                                                  bins=50)

    counts, _, _ = np.histogram2d(filtered_df['longitude'], filtered_df['latitude'], bins=50)
    heatmap_data /= (counts + 1e-10)  # avoid division by zero

    plt.figure(figsize=(12, 8))
    cmap = 'jet' if feature == 'altitude' else 'viridis'  # Different color schemes for altitude and groundspeed
    plt.imshow(heatmap_data.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto',
               cmap=cmap, interpolation='gaussian')

    cbar = plt.colorbar()
    cbar.set_label(f'Average {feature.capitalize()}', rotation=270, labelpad=15)

    plt.title(f'Heatmap of Average {feature.capitalize()} near Frankfurt Airport', pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig(r'C:\Users\saran\Desktop\Plots\heatmap_avg_' + feature + '.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    total_rows = sum(1 for line in open(PATH_DATA)) - 1
    pbar = tqdm(total=total_rows)

    data_list = []
    for chunk in pd.read_csv(PATH_DATA, chunksize=10000, parse_dates=['timestamp']):
        data_list.append(chunk)
        pbar.update(len(chunk))
    pbar.close()
    data = pd.concat(data_list, ignore_index=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print(data.columns)

    plot_origin_density(data)           # Plot : Origin Density
    data = preprocess_plot_data(data)   # Preprocessing
    plot_flight_times_average(data)     # Plot : Average/Median Flight Times
    plot_busy_hours(data)               # Plot : Busy hours
    #plot_gs_alt_100_FRA(data)           # Plot : Density Near Frankfurt
    plot_heatmap_near_FRA(data, feature='altitude')  # Change feature to 'groundspeed' for groundspeed heatmap
    plot_heatmap_near_FRA(data, feature='groundspeed')  # Change feature to 'groundspeed' for groundspeed heatmap
