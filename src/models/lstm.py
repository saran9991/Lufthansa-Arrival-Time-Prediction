import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LeakyReLU, Dropout, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout, Masking
from tensorflow.keras.optimizers import Adam


def build_lstm(
        lr: float = 0.0001,
        n_features: int = 21,
        output_dims: int = 1,
        lstm_layers: tuple = (1024, 512),
        dense_layers: tuple = (512, 256, 128),
        dropout_rate_fc: float = 0.2,
        dropout_rate_lstm: float = 0.2,
        activation: str = "softplus",
        loss: str = "MAE",
        masking_value: float = 0.0,  # Default masking_value set to 0.0; adjust as needed
        use_masking: bool = False  # Parameter to decide whether or not to use the Masking layer
):
    model = Sequential()
    model.add(Input(shape=(None, n_features)))

    if use_masking:
        model.add(Masking(mask_value=masking_value))  # Masking layer to handle padding values if applied

    # Add LSTM layers
    for i in range(len(lstm_layers) - 1):
        model.add(
            LSTM(
                lstm_layers[i],
                return_sequences=True,
                dropout=dropout_rate_lstm,
                #recurrent_dropout=dropout_lstm_recurrent
            ))
    model.add(LSTM(
        lstm_layers[-1],
        return_sequences=False,
        dropout=dropout_rate_lstm,
        #recurrent_dropout=dropout_lstm_recurrent,
    ))  # Last LSTM layer with return_sequences=False

    # Add Dense layers
    for size in dense_layers:
        model.add(Dense(size))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(dropout_rate_fc))

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
    def __init__(self, scaler=None, distance_relative=False, index_distance=1, model_file=None, **network_params):
        if model_file is None:
            self.model = build_lstm(**network_params)
        else:
            self.model = load_model(model_file)
        self.distance_relative = distance_relative
        self.index_distance = index_distance
        self.scaler = scaler

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
            max_epochs = 20,
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
            epochs=max_epochs,
            steps_per_epoch=X_train.shape[0] // batch_size,
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        loss_fn = get_loss(self.model.loss)

        if self.distance_relative:
            inverser_scale = self.scaler.inverse_transform(X[:, -1, 0:5])
            distances = inverser_scale[:, self.index_distance]
            predictions_relative = self.model.predict(X).flatten()
            predictions_absolute = predictions_relative * distances
            # Calculate and return both losses using the retrieved loss function
            loss_relative = float(loss_fn(y, predictions_relative).numpy())
            r2_relative = r2_score(y, predictions_relative)
            loss_absolute = float(loss_fn(y * distances, predictions_absolute).numpy())
            r2_absolute = r2_score(y * distances, predictions_absolute)
            print(f"Evaluation Results:\n"
                  f" - Loss (relative to distance): {loss_relative:.4f}\n"
                  f" - Loss (absolute): {loss_absolute:.4f}\n"
                  f" - R2 (relative to distance): {r2_relative:.4f}\n"
                  f" - R2 (absolute): {r2_absolute:.4f}\n")
            return loss_relative, loss_absolute, r2_relative, r2_absolute

        loss = self.model.evaluate(X, y)
        print(f"Evaluation Result:\n"
              f" - Loss: {loss:.4f}\n")
        return loss, None

    def get_shap(self, arr_train, arr_test, FEATURES, file= "shap_plot.png", title="Feature Importance"):
        """
        Compute and plot SHAP values for a given trained model and datasets.

        Parameters:
        - df_train (pd.DataFrame): The processed training dataset. Should only contain features used for model training.
        - df_test (pd.DataFrame): The processed testing dataset. Should only contain features used for model training.
        - model (trained model): The trained machine learning model. It should be already trained.
        - FEATURES (list): List of feature names.

        Note: Ensure the input datasets (df_train and df_test) are already processed and the model is trained.
        """

        explainer = shap. DeepExplainer(self.model, arr_train)
        shap_values = explainer(arr_test)

        shap.summary_plot(shap_values, arr_test, feature_names=FEATURES, show=False)
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