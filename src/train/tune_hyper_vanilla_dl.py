import re
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from bayes_opt import BayesianOptimization
from src.models.fnn import VanillaNN
from src.processing_utils.preprocessing import seconds_till_arrival

PATH_TRAINING_DATA = os.path.join("..", "..", "data", "processed", "training_data_0617.csv")
PATH_TEST_DATA = os.path.join("..", "..", "data", "processed", "test_data_2023_Jan-Mai.csv")
PATH_SCALER = os.path.join("..", "..", "trained_models", "std_scaler_reg_new.bin")
PATH_MODEL =os.path.join("..", "..", "trained_models", "vanilla_nn")

COLS_NUMERIC = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

FEATURES = ['distance', 'altitude', 'geoaltitude', 'vertical_rate', 'groundspeed', 'holiday', 'sec_sin', 'sec_cos', 'day_sin',
            'day_cos', 'bearing_sin', 'bearing_cos', 'track_sin', 'track_cos', 'latitude_rad', 'longitude_rad',
            'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']

COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]

param_bounds = {
    "batch_size": (32, 700),
    "dropout_rate": (0.05, 5),
    "n_layers": (0.51, 3.49),  # Example range, adjust as per requirement
    "neurons_layer_1": (128, 4096),
    "neurons_layer_2": (128, 4096),
    "neurons_layer_3": (128, 4096),
}


def thin_data(df: pd.DataFrame, target: pd.Series, thinning_factor: int):
    """
    Thins the input dataframe and target series by selecting every
    `thinning_factor`-th row after sorting by 'arrival_time' and 'timestamp'.

    Parameters:
        df (pd.DataFrame): The input dataframe to thin.
        target (pd.Series): The target values corresponding to df.
        thinning_factor (int): The thinning parameter to use.

    Returns:
        pd.DataFrame, pd.Series: Thinned dataframe and target series.
    """
    if thinning_factor <= 0:
        raise ValueError("Thinning factor must be a positive integer.")

    # Ensure the data is sorted by 'arrival_time' and 'timestamp' before thinning
    df_sorted = df.sort_values(by=['arrival_time', 'timestamp'])
    target_sorted = target.loc[df_sorted.index]

    # Use slicing to select every `thinning_factor`-th row
    thinned_df = df_sorted.iloc[::thinning_factor, :]
    thinned_target = target_sorted.iloc[::thinning_factor]

    return thinned_df, thinned_target

def register_params(optimizer, text_file= "optimization vanilla 2.txt"):
    """load known results from previous optimization"""
    pattern = re.compile(r'\|\s*\d+\s*\|\s*[-]*\d+(\.\d+)?(e[+-]?\d+)?')
    with open(text_file, "r") as file:
        for line in file:
            if pattern.search(line):
                try:
                    values = [float(val.strip()) for val in line.split("|")[1:8]]
                    target = values[1]
                    params = {
                        "dropout_rate": values[2],
                        "n_layers": int(values[3]),  # Casting to int as it appears to be an integer parameter
                        "neurons_layer_1": values[4],
                        "neurons_layer_2": values[5],
                        "neurons_layer_3": values[6],
                    }
                    optimizer.register(params=params, target=target)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}")

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


if __name__ == "__main__":
    scaler = load(PATH_SCALER)
    df_flights = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])


    arrival_times_train = df_flights.arrival_time.unique()

    # also split test set into evaluate and optimize sets. Optimize is used for objective function
    # evaluate set is used for teh very last test, after full model is trained.
    path_df_eval = os.path.join("..", "..", "data", "processed", "final_testset_vanilla.csv")
    df_evaluate = pd.read_csv(path_df_eval, parse_dates=["arrival_time", "timestamp"])
    df_test = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    y_test = seconds_till_arrival(df_test)
    #arrival_times_test = df_test.arrival_time.unique()
    #optimize_times = np.random.choice(arrival_times_test, size=int(0.5 * len(arrival_times_test)), replace=False)
    #df_optimize = df_test.loc[df_test.arrival_time.isin(optimize_times)]
    #y_optimize = seconds_till_arrival(df_optimize)
    #df_evaluate = df_test.loc[~df_test.arrival_time.isin(optimize_times)]
    #y_evaluate = seconds_till_arrival(df_evaluate)
    evaluate_times = df_evaluate.arrival_time.unique()
    df_evaluate = df_test.loc[df_test.arrival_time.isin(evaluate_times)]
    y_evaluate = seconds_till_arrival(df_evaluate)
    df_optimize = df_test.loc[~df_test.arrival_time.isin(evaluate_times)]
    y_optimize = seconds_till_arrival(df_optimize)
    #df_evaluate.to_csv(path_df_eval, index=False)

    def objective_function(dropout_rate, n_layers, neurons_layer_1, neurons_layer_2, neurons_layer_3):
        train_times = np.random.choice(arrival_times_train, size=int(0.95 * len(arrival_times_train)), replace=False)
        df_train = df_flights.loc[df_flights.arrival_time.isin(train_times)]
        y_train = seconds_till_arrival(df_train)
        df_val = df_flights.loc[~df_flights.arrival_time.isin(train_times)]
        y_val = seconds_till_arrival(df_val)
        X_train, target_train = thin_data(df_train, y_train, thinning_factor=4)
        layers = (neurons_layer_1, neurons_layer_2, neurons_layer_3)

        layer_sizes = tuple([round(layers[i]) for i in range(round(n_layers))])


        model = VanillaNN(
            features=FEATURES,
            scaler=scaler,
            cols_to_scale=COLS_TO_SCALE,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            distance_relative=True,
        )

        print("Current Params dropout {}, architecture {}".format(dropout_rate, layer_sizes))
        model.fit(X_train, target_train, df_val, y_val, batch_size=256, patience_early=3, patience_reduce=2)
        loss = model.evaluate(df_optimize, y_optimize)
        return -loss[0]


    # Define a regex pattern to find relevant rows


    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=1,
    )
    # Parse the file and register data points
    #register_params(optimizer)

    optimizer.maximize(n_iter=10, init_points=5)
    best_params = optimizer.max['params']
    best_target = optimizer.max['target']

    # Displaying the best parameters
    print("The optimal parameters are: {}".format(best_params))
    print("The maximum value of the target function is: {}".format(best_target))
    # Access all results
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
