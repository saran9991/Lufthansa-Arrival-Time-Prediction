import copy
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from bayes_opt import BayesianOptimization
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "processed", "test_data.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_reg_new.bin")
PATH_MODEL =os.path.join("..", "..", "trained_models", "vanilla_nn")

COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

FEATURES = ['distance', 'altitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
            'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
            'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

param_bounds = {
    'exponent': (5, 12),  # [2^5, 2^11] -> [32, 2048]
    'n_layers': (1, 6),  # Example range, adjust as per requirement
    'dropout_rate': (0, 0.8),
}

def calc_layers(exponent, n_layers):
    initial_neurons = 2 ** round(exponent)  # Calculate initial neuron count
    layers = [initial_neurons]  # List to hold the neuron counts for each layer

    # Add subsequent layers, halving neuron count each time
    for _ in range(1, round(n_layers)):
        next_layer_neurons = layers[-1] // 2  # Halve the neuron count

        # Check if the neuron count is below the minimum (2)
        if next_layer_neurons < 2:
            break

        layers.append(next_layer_neurons)

    return tuple(layers)



if __name__ == "__main__":
    scaler = load(PATH_SCALER)
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])

    # Split unique arrival_time 90/10
    arrival_times = df_flights.arrival_time.unique()
    train_times = np.random.choice(arrival_times, size=int(0.95 * len(arrival_times)), replace=False)

    df_train = df_flights.loc[df_flights.arrival_time.isin(train_times)]
    y_train = seconds_till_arrival(df_train)
    df_val = df_flights.loc[~df_flights.arrival_time.isin(train_times)]
    y_val = seconds_till_arrival(df_val)

    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_test = seconds_till_arrival(df_test)

    def objective_function(exponent, n_layers, dropout_rate):

        layer_sizes = calc_layers(exponent, n_layers)


        model = VanillaNN(
            features=FEATURES,
            scaler=scaler,
            cols_to_scale=COLS_TO_SCALE,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            distance_relative=True,
        )
        print("Current Params dropout {}, architecture {}".format(dropout_rate, layer_sizes))
        model.fit(df_train, y_train, df_val, y_val, batch_size=512, patience_early=1, patience_reduce=1)
        loss = model.evaluate(df_test, y_test)
        return -loss[0]


    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=1,
    )
    optimizer.maximize(n_iter=10)
    # Access all results
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    #model.model.save(PATH_MODEL)

