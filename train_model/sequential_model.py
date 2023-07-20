import copy
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np



now = datetime.now()
dt_string = now.strftime("%y%m%d_%H_%M_%S")

class SequentialModel:

    def __init__(
            self,
            build_new = True,
            model_path = "../trained_models/sequential_model_" + dt_string,
            params: dict = {},
            scaler = None,
            cols_to_scale = None,
            postprocess_output_function = None,
            model_type = "vanilla"
    ):
        if build_new:
            if model_type == "lstm":
                build_function = build_lstm
            else:
                build_function = build_sequential
            self.model = build_function(
                lr=params["lr"],
                input_dims = params["input_dims"],
                output_dims=params["output_dims"],
                layer_sizes = params["layer_sizes"],
                dropout_rate = params["dropout_rate"],
                activation = params["activation"],
                loss = params["loss"],
            )

        else:
            self.model = load_model(model_path)
        self.model_path = model_path
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale
        self.postprocess_output_function = postprocess_output_function


    def change_learning_rate(self, lr):
        K.set_value(self.model.optimizer.learning_rate, lr)

    def save_model(self, path = None):
        if path == None:
            path = self.model_path
        self.model.save(path)

    def load_model(self, path = None):
        if path == None:
            path = self.model_path
        self.model = load_model(path)

    def predict(self, X, postprocess_output=False):
        if self.scaler is not None:
            X_predict = copy.deepcopy(X)
            X_predict.iloc[:, self.cols_to_scale] = self.scaler.transform(X.iloc[:, self.cols_to_scale])
        else:
            X_predict = X
        prediction = self.model.predict(X_predict)
        if postprocess_output:
            prediction = np.array(prediction)

            return self.postprocess_output_function(X, prediction)

        return prediction

    def evaluate(self, X, y,postprocess_output=False):
        if self.scaler is not None:
            X_eval = copy.deepcopy(X)
            X_eval.iloc[:, self.cols_to_scale] = self.scaler.transform(X.iloc[:, self.cols_to_scale])
        else:
            X_eval = X

        if postprocess_output:
            y_pred = self.predict(X, postprocess_output=True)
            y_pred = y_pred.reshape(-1)
            mae = (np.abs(y_pred-y)).mean()
            print("mae:", mae)
            return mae

        else:
            return self.model.evaluate(X_eval, y)



def build_sequential(lr, input_dims, output_dims, layer_sizes, dropout_rate, activation, loss):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_dims))
    for size in layer_sizes:
        model.add(keras.layers.Dense(size))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dropout(dropout_rate))  # Add dropout layer here

    model.add(keras.layers.Dense(output_dims,activation=activation))

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)

    return model

def build_lstm(lr, input_dims, output_dims, layer_sizes, dropout_rate, lstm_layers, activation, loss):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(None, input_dims)))  # Assuming timesteps is variable (None)
    for i in range(lstm_layers - 1):
        model.add(keras.layers.LSTM(layer_sizes[i], return_sequences=True))  # Add LSTM layer here with return_sequences=True
    model.add(keras.layers.LSTM(layer_sizes[lstm_layers - 1], return_sequences=False))  # Last LSTM layer with return_sequences=False

    for size in layer_sizes[lstm_layers:]:
        model.add(keras.layers.Dense(size))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dropout(dropout_rate))  # Add dropout layer here

    model.add(keras.layers.Dense(output_dims, activation=activation))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

    return model
