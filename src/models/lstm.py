import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LeakyReLU, Dropout, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm(
        lr: float = 0.0001,
        n_features: int = 21,
        output_dims: int = 1,
        lstm_layers: tuple = (1024, 512),
        dense_layers: tuple = (512, 256, 128),
        dropout_rate: float = 0.2,
        activation: str = "softplus",
        loss: str = "MAE",
):
    model = Sequential()
    model.add(Input(shape=(None, n_features)))
    for i in range(len(lstm_layers) - 1):
        model.add(LSTM(lstm_layers[i], return_sequences=True))
    model.add(LSTM(lstm_layers[- 1], return_sequences=False))  # Last LSTM layer with return_sequences=False

    for size in dense_layers:
        model.add(Dense(size))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_dims, activation=activation))

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)

    return model

def batch_generator(X, y, batchsize):
    """X must be 3-d np-arrays of timeseries, y is a 1-d array of targets. The data is already preprocessed"""
    size = X.shape[0]
    while True:
        shuffled_indices = np.random.permutation(np.arange(size))
        X = X[shuffled_indices, :, :]
        y = y[shuffled_indices]

        i = 0
        while i < size:
            X_batch = X[i:i+batchsize, :, :]
            y_batch = y[i:i+batchsize]

            yield X_batch, y_batch
            i += batchsize


        yield X_batch, y_batch

class LSTMNN():
    def __init__(self, model_file=None, **network_params):
        if model_file is None:
            self.model = build_lstm(**network_params)
        else:
            self.model = load_model(model_file)

    def fit(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            patience_early=7,
            patience_reduce=3,
            reduce_factor=0.7,
            batch_size=32,
    ):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience_early,
            verbose=1,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_factor,
            patience=patience_reduce,
            min_lr=1e-7
        )
        generator = batch_generator(X_train, y_train, batchsize=batch_size)
        self.model.fit(
            generator,
            max_queue_size=2000,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            epochs=2000,
            steps_per_epoch=X_train.shape[0] // batch_size,
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)