import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from joblib import dump, load
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LeakyReLU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import get as get_loss
import tensorflow as tf
from src.processing_utils.preprocessing import generate_aux_columns

gpus = tf.config.list_physical_devices('GPU')
print("Number of GPUs available:", len(gpus))

def build_sequ


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
            std_scaler=None,
            minmax_scaler=None,
            cols_to_scale_std: list = None,
            cols_to_scale_minmax: list = None,
            model_file: str = None,
            distance_relative: bool = False,
            save_std_scaler_file: str = None,
            save_minmax_scaler_file: str = None,
            **network_params
    ):
        if model_file is None:
            input_dims = (len(features),)
            self.model = build_sequential(input_dims=input_dims, **network_params)
        else:
            self.model = load_model(model_file)

        self.feature_columns = features
        self.std_scaler = std_scaler
        self.minmax_scaler = minmax_scaler
        self.std_scaler_file = save_std_scaler_file
        self.minmax_scaler_file = save_minmax_scaler_file
        self.cols_to_scale_std = cols_to_scale_std
        self.cols_to_scale_minmax = cols_to_scale_minmax
        self.distance_relative = distance_relative

    def preprocess(self, features):
        if self.std_scaler is None:
            self.std_scaler = StandardScaler()
            self.std_scaler.fit(features[self.cols_to_scale_std])
            dump(self.std_scaler, self.std_scaler_file)
        if self.minmax_scaler is None:
            self.minmax_scaler= MinMaxScaler()
            self.minmax_scaler.fit(features[self.cols_to_scale_minmax])
            dump(self.minmax_scaler, self.minmax_scaler_file)
        X = features.copy()
        X = generate_aux_columns(X)
        X[self.cols_to_scale_std] = self.std_scaler.transform(X[self.cols_to_scale_std])
        X[self.cols_to_scale_minmax] = self.minmax_scaler.transform(X[self.cols_to_scale_minmax])
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
            max_epochs=500,
    ):
        if self.distance_relative:
            y_train = y_train/X_train["distance"]
            y_val = y_val/X_val["distance"]
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
            epochs=max_epochs,
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
            X_unscaled = self.std_scaler.inverse_transform(X[self.cols_to_scale_std])

            if isinstance(X, pd.DataFrame):
                distances = X_unscaled[:, self.cols_to_scale_std.index("distance")]
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
            r2_relative = r2_score(y_test / distances, predictions_relative)
            r2_absolute = r2_score(y_test, predictions_absolute)
            print(f"Evaluation Results:\n"
                  f" - Loss (relative to distance): {loss_relative:.4f}\n"
                  f" - Loss (absolute): {loss_absolute:.4f}\n"
                  f" - R^2 (relative to distance): {r2_relative:.4f}\n"
                  f" - R^2 (absolute): {r2_absolute:.4f}\n")
            return loss_relative, loss_absolute, r2_relative, r2_absolute


        # If the model is trained to predict absolute times till arrival
        else:
            # Predict using the model
            predictions = self.model.predict(features_test, batch_size=batch_size).flatten()

            # Calculate and return the loss using the retrieved loss function
            loss = float(loss_fn(y_test, predictions).numpy())
            r2 = r2_score(y_test, predictions)
            print(f"Evaluation Result:\n"
                  f" - Loss: {loss:.4f}\n"
                  f" - R^2: {r2:.4f}\n")
            return loss, None, r2, None



    def get_shap(self, df_train, df_test, FEATURES = None, file= "shap_plot.png", title="Feature Importance"):
        """
        Compute and plot SHAP values for a given trained model and datasets.

        Parameters:
        - df_train (pd.DataFrame): The processed training dataset. Should only contain features used for model training.
        - df_test (pd.DataFrame): The processed testing dataset. Should only contain features used for model training.
        - model (trained model): The trained machine learning model. It should be already trained.
        - FEATURES (list): List of feature names.

        Note: Ensure the input datasets (df_train and df_test) are already processed and the model is trained.
        """
        df_train_processed = self.preprocess(df_train)
        df_test_processed = self.preprocess(df_test)
        df_train_processed= df_train_processed.to_numpy()
        df_test_processed = df_test_processed.to_numpy()

        explainer = shap.GradientExplainer(self.model, df_train_processed)
        shap_values = explainer(df_test_processed)
        if FEATURES is None:
            FEATURES = self.feature_columns
        shap.summary_plot(shap_values, df_test_processed, feature_names=FEATURES, show=False)
        ax = plt.gca()  # Get current axis
        ax.set_xlabel("Mean Absolute Shap Value")
        # Remove any legends
        if ax.get_legend():
            ax.get_legend().remove()
        bars = ax.patches  # Get the bars from the axis

        # Modify the color of each bar
        for bar in bars:
            bar.set_facecolor("#76C1C1")
        # Clear x-axis label if it has the unwanted "class" text
        if "class" in ax.get_xlabel():
            ax.set_xlabel('')
        # Save the plot
        plt.savefig(file, bbox_inches='tight', dpi=300)
        plt.close()  # Optional: close the figure after saving