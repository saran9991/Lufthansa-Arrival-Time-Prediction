import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LeakyReLU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import get as get_loss
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
    def __init__(
            self,
            features: list,
            scaler=None,
            cols_to_scale: list = None,
            model_file: str = None,
            distance_relative: bool = False,
            **network_params
    ):
        if model_file is None:
            input_dims = (len(features),)
            self.model = build_sequential(input_dims=input_dims, **network_params)
        else:
            self.model = load_model(model_file)

        self.feature_columns = features
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale
        self.distance_relative = distance_relative

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
        if self.distance_relative:
            y_train = y_train/X_train.distance
            y_val = y_val/X_val.distance
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
        # Initialize callbacks list with always-used callbacks
        callbacks = [early_stopping, reduce_lr]


        generator = batch_generator(features_train, y_train, batchsize=batch_size)


        self.model.fit(
            generator,
            max_queue_size=2000,
            validation_data=(features_val, y_val),
            callbacks=callbacks,
            epochs=2000,
            steps_per_epoch=X_train.shape[0] // batch_size,
        )

    def predict(self, X, preprocess=True, index_distance=0):
        # If preprocessing is required
        if preprocess:
            X_pred = self.preprocess(X)

            # If predictions should be relative to distance
            if self.distance_relative:
                if isinstance(X, pd.DataFrame):
                    distances = X["distance"].values
                else:
                    distance_index = self.feature_columns.index("distance")
                    distances = X[:, distance_index]

                # Ensure distances is 2D for broadcasting
                distances = distances.reshape(-1, 1)
                predictions = self.model.predict(X_pred) * distances
                return predictions

            # If no distance relation is required
            return self.model.predict(X_pred)

        # If no preprocessing is required but predictions should be relative to distance
        elif not preprocess and self.distance_relative:
            X_unscaled = self.scaler.inverse_transform(X[self.cols_to_scale])

            if isinstance(X, pd.DataFrame):
                distances = X_unscaled[:, self.cols_to_scale.index("distance")]
            else:
                distances = X_unscaled[:, index_distance]

            # Ensure distances is 2D for broadcasting
            distances = distances.reshape(-1, 1)
            return self.model.predict(X) * distances

        # If no preprocessing and no distance relation is required
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate the model on a test set. Return the loss for the raw predictions
        and, if self.distance_relative is True, the loss for the predictions converted
        back to seconds till arrival by multiplying with the distance.
        """
        # Ensure the features are preprocessed
        features_test = self.preprocess(X_test)

        # Retrieve the actual loss function object from the string
        loss_fn = get_loss(self.model.loss)

        # Get distances, assuming X_test is a DataFrame
        distances = X_test["distance"].values

        # If the model is trained to predict relative times till arrival
        if self.distance_relative:
            # Predict using the model
            predictions_relative = self.model.predict(features_test, batch_size=batch_size).flatten()

            # Convert the relative predictions back into seconds till arrival
            predictions_absolute = predictions_relative * distances

            # Calculate and return both losses using the retrieved loss function
            loss_relative = float(loss_fn(y_test / distances, predictions_relative).numpy())
            loss_absolute = float(loss_fn(y_test, predictions_absolute).numpy())
            print(f"Evaluation Results:\n"
                  f" - Loss (relative to distance): {loss_relative:.4f}\n"
                  f" - Loss (absolute): {loss_absolute:.4f}\n")
            return loss_relative, loss_absolute


        # If the model is trained to predict absolute times till arrival
        else:
            # Predict using the model
            predictions = self.model.predict(features_test, batch_size=batch_size).flatten()

            # Calculate and return the loss using the retrieved loss function
            loss = float(loss_fn(y_test, predictions).numpy())
            print(f"Evaluation Result:\n"
                  f" - Loss: {loss:.4f}\n")

            return loss, None