import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%y%m%d_%H_%M_%S")

class SequentialModel:

    def __init__(
            self,
            build_new = True,
            model_path = "../trained_models/sequential_model_" + dt_string,
            params: dict = {}
    ):
        if build_new:
            self.model = build_sequential(
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
