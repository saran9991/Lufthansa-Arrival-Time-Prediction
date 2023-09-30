import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LeakyReLU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.processing_utils.preprocessing import generate_aux_columns


def build_sequential(
        lr: float = 0.001,
        input_dims: tuple = (21,),
        output_dims: int = 1,
        layer_sizes: tuple = (1024, 512, 256),
        dropout_rate: float = 0.2,
        activation: str = "softplus",
        loss: str = "MAE",
):
    model = Sequential()
    model.add(Input(shape=input_dims))
    for size in layer_sizes:
        model.add(Dense(size))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_dims, activation=activation))
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    return model


def batch_generator(X, y, batchsize):
    size = X.shape[0]
    while True:
        shuffled_indices = np.random.permutation(np.arange(size))
        X = X.iloc[shuffled_indices, :]
        y = y.iloc[shuffled_indices]

        i = 0
        while i < size:
            X_batch = X.iloc[i:i + batchsize, :]
            y_batch = y.iloc[i:i + batchsize].values

            yield X_batch, y_batch
            i += batchsize

        X_batch = X.iloc[i:, :]
        y_batch = y.iloc[i:].values
        yield X_batch, y_batch


class VanillaNN():
    def __init__(self, features, scaler=None, cols_to_scale=None, model_file=None, **network_params):
        if model_file is None:
            self.model = build_sequential(**network_params)
        else:
            self.model = load_model(model_file)

        self.feature_columns = features
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale

    def preprocess(self, features):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(features[self.cols_to_scale])
        X = features.copy()
        X = generate_aux_columns(X)
        X[self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])
        return X[self.feature_columns]

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
        features_train = self.preprocess(X_train)
        features_val = self.preprocess(X_val)

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
        generator = batch_generator(features_train, y_train, batchsize=batch_size)

        self.model.fit(
            generator,
            max_queue_size=2000,
            validation_data=(features_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            epochs=2000,
            steps_per_epoch=X_train.shape[0] // batch_size,
        )

    def predict(self, X, preprocess=True):
        if preprocess:
            X_pred = self.preprocess(X)
            return self.model.predict(X_pred)
        else:
            return self.model.predict(X)

    def evaluate(self, X, y, preprocess=True):
        if preprocess:
            X_eval = self.preprocess(X)
            self.model.evaluate(X_eval, y)
        else:
            self.model.evaluate(X, y)