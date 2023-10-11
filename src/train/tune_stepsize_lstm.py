import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from joblib import load as load_joblib
from bayes_opt import BayesianOptimization
from src.models.lstm import LSTMNN
from src.processing_utils.create_timeseries import create_time_series_array
from src.processing_utils.preprocessing import generate_aux_columns, seconds_till_arrival


PATH_DATA = os.path.join("..", "..", "data", "processed")
PATH_TRAINING_DATA = os.path.join(PATH_DATA, "training_data_2022_10sec_sample.csv")
PATH_TEST_DATA = os.path.join(PATH_DATA, "training_data_2023_10sec_sample.csv")
PATH_MODEL = os.path.join("..", "..", "trained_models", "lstm_near_231003")
scaler_path = os.path.join("../..", "trained_models", "std_scaler_reg_new.bin")
drop_columns = ["timestamp", "track", "latitude", "longitude", "arrival_time"]
COLS_TO_SCALE = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
scaler = load_joblib(scaler_path)

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

def create_training_data(train = True, test_size=0.2, n_steps= 40, stepsize=1):
    stride_train = max(1, round(10/stepsize))
    stride_val = max(1, round(25 / stepsize))
    if train:
        df = pd.read_csv(PATH_TRAINING_DATA, parse_dates=["arrival_time", "timestamp"])
    else:
        df = pd.read_csv(PATH_TEST_DATA, parse_dates=["arrival_time", "timestamp"])
    arrival_days = df['arrival_time'].dt.date.unique()
    df['flight_id'] = df['flight_id'].astype(str) + '_' + df['arrival_time'].dt.date.astype(str)

    if train:
        arrival_days_train, arrival_days_val = train_test_split(arrival_days, test_size=test_size, random_state=42)
        # Identify the flight_ids that correspond to each day of arrival
        flight_ids_train = df[df['arrival_time'].dt.date.isin(arrival_days_train)]['flight_id'].unique()
        flight_ids_val = df[df['arrival_time'].dt.date.isin(arrival_days_val)]['flight_id'].unique()
    else:
        flight_ids_train = df['flight_id'].unique()
    df = generate_aux_columns(df)
    y = seconds_till_arrival(df)
    df = df.drop(columns=drop_columns)

    X_numeric = df[COLS_TO_SCALE]
    df[COLS_TO_SCALE] = scaler.transform(X_numeric)
    df["time_to_arrival"] = y.values

    time_series_array = create_time_series_array(
        df,
        n_steps,
        flight_ids_train,
        apply_padding=False,
        stride=stride_train,
        stepsize=stepsize
    )


    X = time_series_array[:, :, :-1]  # last column is the time_to_arrival
    y = time_series_array[:, -1, -1]  # the last y-value in the sequence is the target
    if train:
        time_series_array_val = create_time_series_array(
            df,
            n_steps,
            flight_ids_val,
            apply_padding=False,
            stride=stride_val,
            stepsize=stepsize
        )
        X_val = time_series_array_val[:, :, :-1]
        y_val = time_series_array_val[:, -1, -1]
        print("shape train", X.shape, "shape val", X_val.shape)
        return X, y, X_val, y_val

    print("shape array", X.shape)
    return X, y



if __name__ == "__main__":
    param_bounds = {
        'exponent_lstm': (7, 13),  # [2^5, 2^11] -> [32, 2048]
        'n_layers_lstm': (1, 6),  # Example range, adjust as per requirement
        'exponent_fc': (8, 13),  # [2^5, 2^11] -> [32, 2048]
        'n_layers_fc': (1, 6),  # Example range, adjust as per requirement
        'dropout_rate_fc': (0, 0.8),
        'stepsize': (1,10),
        'sequence_length': (4, 50),

    }
    def objective_function(
            exponent_lstm,
            n_layers_lstm,
            exponent_fc,
            n_layers_fc,
            dropout_rate_fc,
            stepsize,
            sequence_length,
    ):
        stepsize = round(stepsize)
        sequence_length = round(sequence_length)

        X_train, y_train, X_val, y_val = create_training_data(
            train=True,
            test_size=0.2,
            n_steps=sequence_length,
            stepsize=stepsize
        )

        layer_sizes_lstm = calc_layers(exponent_lstm, n_layers_lstm)
        layer_sizes_fc = calc_layers(exponent_fc, n_layers_fc)
        n_features = X_train.shape[2]
        model = LSTMNN(
            n_features=n_features,
            lr=0.001,
            lstm_layers=layer_sizes_lstm,
            dense_layers=layer_sizes_fc,
            dropout_rate=dropout_rate_fc,
        )
        print("""
        Current Params: 
        dropout fc {}, architecture fc {}, 
        architecture lstm{}, stepsize: {}, sequence length: {}
        """.format(
            dropout_rate_fc,
            layer_sizes_fc,
            layer_sizes_lstm,
            stepsize,
            sequence_length
        ))
        model.fit(X_train, y_train, X_val, y_val, batch_size=256, patience_early=2, patience_reduce=1)

        X_test, y_test = create_training_data(
            train=False,
            n_steps=sequence_length,
            stepsize=stepsize
        )
        loss = model.evaluate(X_test, y_test)
        return -loss

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=1,
    )
    optimizer.maximize(n_iter=20)
    # Access all results
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))







