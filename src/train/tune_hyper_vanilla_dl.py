import re
import math
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from bayes_opt import BayesianOptimization
import logging
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "final", "train", "training_data_2022_100km.csv")
scaler_path_std = os.path.join("..", "..", "trained_models", "std_scaler_100km_h3.bin")
scaler_path_minmax = os.path.join("..", "..", "trained_models", "minmax_scaler_100km_h3.bin")


COLS_TO_SCALE_STD = [
    "distance",
    "altitude",
    "geoaltitude",
    "vertical_rate",
    "groundspeed",
]
COLS_TO_SCALE_MINMAX = [
    "density_10_minutes_past",
    "density_30_minutes_past",
    "density_60_minutes_past",
]


FEATURES = ['distance',
            'altitude',
            'geoaltitude',
            'vertical_rate',
            'groundspeed',
            'holiday',
            'sec_sin',
            'sec_cos',
            'day_sin',
            'day_cos',
            'bearing_sin',
            'bearing_cos',
            'track_sin',
            'track_cos',
            'latitude_rad',
            'longitude_rad',
            'weekday_1',
            'weekday_2',
            'weekday_3',
            'weekday_4',
            'weekday_5',
            'weekday_6',
            "density_10_minutes_past",
            "density_30_minutes_past",
            "density_30_minutes_past",
            ]

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]


param_bounds = {
    "lr_start": (-10, -4),  # take exp
    "batch_size": (5, 8),  # exp base two
    "dropout_rate": (0.05, 0.6),
    "n_layers": (0.51, 3.49),  # Example range, adjust as per requirement
    "neurons_layer_1": (7, 12),  # exp base two
    "neurons_layer_2": (7, 12),
    "neurons_layer_3": (7, 12),
    #"patience_reduce": (0.51, 5.49),
}

def register_params(optimizer, text_file= "optimization vanilla.txt"):
    """load known results from previous optimization"""
    pattern = re.compile(r'\|\s*\d+\s*\|\s*[-]*\d+(\.\d+)?(e[+-]?\d+)?')
    with open(text_file, "r") as file:
        for line in file:
            if pattern.search(line):
                try:
                    values = [float(val.strip()) for val in line.split("|")[1:11]]
                    target = values[1]
                    params = {
                        "batch_size" : values[2],
                        "dropout_rate": values[3],
                        "lr_start": values[4],
                        "n_layers": int(values[5]),  # Casting to int as it appears to be an integer parameter
                        "neurons_layer_1": values[6],
                        "neurons_layer_2": values[7],
                        "neurons_layer_3": values[8],
                        "patience_reduce": values[9],
                    }
                    optimizer.register(params=params, target=target)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}")

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


if __name__ == "__main__":
    std_scaler = load(scaler_path_std)
    minmax_scaler = load(scaler_path_minmax)
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    df_flights["seconds_till_arrival"] = seconds_till_arrival(df_flights)
    flight_ids = df_flights.flight_id.unique()

    def objective_function(
            lr_start,
            batch_size,
            dropout_rate,
            n_layers,
            neurons_layer_1,
            neurons_layer_2,
            neurons_layer_3,
            #patience_reduce
    ):
        lr_start = math.exp(lr_start)
        batch_size = round(2**batch_size)
        layers = (2**neurons_layer_1, 2**neurons_layer_2, 2**neurons_layer_3)
        print(layers)
        layer_sizes = tuple([round(layers[i]) for i in range(round(n_layers))])
        patience_reduce = 2
        patience_early = patience_reduce + 1
        unique_flight_ids = df_flights['flight_id'].unique()

        # Shuffle the unique flight_ids
        np.random.shuffle(unique_flight_ids)

        # Calculate split sizes
        total_flights = len(unique_flight_ids)
        train_size = int(0.80 * total_flights)
        valid_size = int(0.05 * total_flights)

        # Split the flight_ids into three parts
        train_flight_ids = unique_flight_ids[:round(train_size/3)] # to speed up hyperparameter tuning
        valid_flight_ids = unique_flight_ids[train_size:train_size + valid_size]
        test_flight_ids = unique_flight_ids[train_size + valid_size:]

        # Create three dataframes
        df_train = df_flights[df_flights['flight_id'].isin(train_flight_ids)]
        X_train = df_train.drop(columns="seconds_till_arrival")
        y_train = df_train.seconds_till_arrival
        df_val = df_flights[df_flights['flight_id'].isin(valid_flight_ids)]
        X_val= df_val.drop(columns="seconds_till_arrival")
        y_val = df_val.seconds_till_arrival
        df_test = df_flights[df_flights['flight_id'].isin(test_flight_ids)]
        X_test= df_test.drop(columns="seconds_till_arrival")
        y_test = df_test.seconds_till_arrival
        # Logging the shapes
        logger.info(f"Total data shape: {df_flights.shape}")
        logger.info(f"Training data shape: {df_train.shape}")
        logger.info(f"Validation data shape: {df_val.shape}")
        logger.info(f"Test data shape: {df_test.shape}")


        model = VanillaNN(
            features=FEATURES,
            std_scaler=std_scaler,
            cols_to_scale_std=COLS_TO_SCALE_STD,
            minmax_scaler=minmax_scaler,
            cols_to_scale_minmax=COLS_TO_SCALE_MINMAX,
            lr=lr_start,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            distance_relative=True,
        )
        input_shape = model.model.layers[0].input_shape
        logging.info('Input shape of the model: %s', input_shape)
        model.model.summary()
        logger.info("Current Params batchsize %s dropout %s, architecture %s, patience %s",
                    batch_size, dropout_rate, layer_sizes, patience_reduce)
        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=round(batch_size),
            patience_early=patience_early,
            patience_reduce=patience_reduce,
            max_epochs=25
        )
        loss = model.evaluate(X_test, y_test)
        return -loss[0]


    # Define a regex pattern to find relevant rows


    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=1,
    )
    # Parse the file and register data points
    #register_params(optimizer)

    optimizer.maximize(n_iter=50, init_points=5)
    best_params = optimizer.max['params']
    best_target = optimizer.max['target']

    # Displaying the best parameters
    print("The optimal parameters are: {}".format(best_params))
    print("The maximum value of the target function is: {}".format(best_target))
    # Access all results
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

