from contextlib import contextmanager
from joblib import load as load_scaler
import logging
import math
import numpy as np
import os
import re
import sys
from bayes_opt import BayesianOptimization
from src.models.lstm import LSTMNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PATH_DATA = os.path.join("..", "..", "data", "final", "train")

PATH_TRAINING_DATA = os.path.join(PATH_DATA, "timeseries_10sec_2022_100km_train_clean.npy")
PATH_VALIDATION_DATA = os.path.join(PATH_DATA, "timeseries_10sec_2022_100km_val_clean.npy")
PATH_OPTIMIZATION_DATA = os.path.join(PATH_DATA, "timeseries_10sec_2022_100km_optim_clean.npy")
PATH_STD_SCALER = os.path.join("..","..", "trained_models", "std_scaler_100km_h3.bin")
print(PATH_VALIDATION_DATA)
@contextmanager
def tee_stdout_to_file(filename):
    original_stdout = sys.stdout

    class Tee:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    with open(filename, 'w', encoding='utf-8') as file:
        sys.stdout = Tee(file, original_stdout)
        yield
        sys.stdout = original_stdout

def register_params(optimizer, text_file= "output.txt"):
    """load known results from previous optimization"""
    pattern = re.compile(r'\|\s*\d+\s*\|\s*[-]*\d+(\.\d+)?(e[+-]?\d+)?')
    with open(text_file, "r") as file:
        for line in file:
            if pattern.search(line):
                try:
                    values = [float(val.strip()) for val in line.split("|")[1:15]]
                    target = values[1]
                    params = {
                        "lr_start": values[2],
                        "batch_size": values[3],
                        "dropout_rate_fc": values[4],
                        "dropout_rate_lstm": values[5],
                        "n_layers_fc": values[6],
                        "n_layers_lstm": values[7],
                        "neurons_layer_1_fc": values[8],
                        "neurons_layer_2_fc": values[9],
                        "neurons_layer_3_fc": values[10],
                        "neurons_layer_1_lstm": values[11],
                        "neurons_layer_2_lstm": values[12],
                        "patience_reduce": values[13],
                    }
                    optimizer.register(params=params, target=target)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}")

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

if __name__ == "__main__":
    param_bounds = {
        "lr_start": (-10, -4),  # take exp
        "batch_size": (5, 10), # exp base two
        "dropout_rate_fc": (0.05, 0.6),
        "dropout_rate_lstm": (0.05, 0.6),
        "n_layers_fc": (0.51, 3.49),  # Example range, adjust as per requirement
        "n_layers_lstm": (0.51, 2.49),
        "neurons_layer_1_fc": (7, 12),  # exp base two
        "neurons_layer_2_fc": (7, 12),
        "neurons_layer_3_fc": (7, 12),
        "neurons_layer_1_lstm": (7, 10),  # exp base two
        "neurons_layer_2_lstm": (7, 9),
        "patience_reduce": (0.51, 5.49),
    }
    scaler = load_scaler(PATH_STD_SCALER)
    data_train = np.load(PATH_TRAINING_DATA)
    data_val = np.load(PATH_VALIDATION_DATA)
    data_optim = np.load(PATH_OPTIMIZATION_DATA)
    X_train = data_train[:, :, :-1] # last column is the time_to_arrival
    y_train = data_train[:, -1, -1] # the last y-value in the sequence is the target
    X_val = data_val[:, :, :-1]
    y_val = data_val[:, -1, -1]
    X_optim = data_optim[:, :, :-1]
    y_optim = data_optim[:, -1, -1]
    logger.info(f"Training data shape: {X_train.shape, y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape, y_val.shape}")
    logger.info(f"Test data shape: {X_optim.shape, y_optim.shape}")

    def objective_function(
        lr_start,
        batch_size,
        dropout_rate_fc,
        dropout_rate_lstm,
        n_layers_fc,
        n_layers_lstm,
        neurons_layer_1_fc,
        neurons_layer_2_fc,
        neurons_layer_3_fc,
        neurons_layer_1_lstm,
        neurons_layer_2_lstm,
        patience_reduce,
    ):
        lr_start = math.exp(lr_start)
        batch_size = round(2**batch_size)

        dropout_rate_fc = dropout_rate_fc
        dropout_rate_lstm = dropout_rate_lstm


        layers_fc = (2**neurons_layer_1_fc, 2**neurons_layer_2_fc, 2**neurons_layer_3_fc)
        print(layers_fc)
        layer_sizes_fc = tuple([round(layers_fc[i]) for i in range(round(n_layers_fc))])
        layers_lstm = (2**neurons_layer_1_lstm, 2**neurons_layer_2_lstm)
        print(layers_lstm)
        layer_sizes_lstm = tuple([round(layers_lstm[i]) for i in range(round(n_layers_lstm))])

        patience_reduce = round(patience_reduce)
        patience_early = patience_reduce + 1



        n_features = X_train.shape[2]
        model = LSTMNN(
            scaler=scaler,
            distance_relative=True,
            index_distance=0,
            n_features=n_features,
            lr=lr_start,
            lstm_layers=layer_sizes_lstm,
            dense_layers=layer_sizes_fc,
            dropout_rate_fc= dropout_rate_fc,
            dropout_rate_lstm=dropout_rate_lstm,
        )
        input_shape = model.model.layers[0].input_shape
        logging.info('Input shape of the model: %s', input_shape)
        model.model.summary()

        logger.info(
            f"Current Params - batchsize: {batch_size}, dropout fc: {dropout_rate_fc}, "
            f"dropout lstm: {dropout_rate_lstm}, layers fc: {layer_sizes_fc}, "
            f"layers lstm: {layer_sizes_lstm}, patience: {patience_reduce}"
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=batch_size,
            patience_early=patience_early,
            patience_reduce=patience_reduce,
        )

        loss = model.evaluate(X_optim, y_optim)
        return -loss[0]

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=1,
    )
    register_params(optimizer, "hyper_lstm_100km.txt")
    # Use the context manager to redirect output of this specific line

    optimizer.maximize(init_points=0, n_iter=41)

    # Access all results
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))







