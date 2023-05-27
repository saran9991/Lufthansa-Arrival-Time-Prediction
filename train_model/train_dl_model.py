from tensorflow import keras, config as tf_config
from multiprocessing import Queue
from joblib import dump, load as load_scaler
from data_loader import load_data
from sequential_model import SequentialModel
import pandas as pd
import os
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def batch_generator(df: pd.DataFrame, y, batchsize, with_sample_weights = False, sample_weights=None):
    size = df.shape[0]
    while True:
        shuffled_indices = np.random.permutation(np.arange(size))
        df = df.iloc[shuffled_indices]
        y = y.iloc[shuffled_indices]

        if with_sample_weights:
            sample_weights = sample_weights.iloc[shuffled_indices]

        i = 0
        while i < size:
            X_batch = df.iloc[i:i+batchsize,:]
            y_batch = y.iloc[i:i+batchsize].values
            if with_sample_weights:
                sample_batch = sample_weights.iloc[i:i+batchsize].values
                yield X_batch, y_batch, sample_batch
            else:
                yield X_batch, y_batch
            i += batchsize

        X_batch = df.iloc[i:,:]
        y_batch = y.iloc[i:].values
        if with_sample_weights:
            sample_batch = sample_weights.iloc[i:].values
            yield X_batch, y_batch, sample_batch
        else:
            yield X_batch, y_batch



if __name__ == "__main__":
    data_files = []
    for i in range(1, 13):
        month = "0" + str(i) if i < 10 else str(i)
        file = ".." + os.sep + "data" + os.sep + "Frankfurt_LH_22" + month + ".h5"
        data_files.append(file)
    data_files_test = [".." + os.sep + "data" + os.sep + "Frankfurt_LH_2301" + ".h5",
                       ".." + os.sep + "data" + os.sep + "Frankfurt_LH_2302" + ".h5",
                       ".." + os.sep + "data" + os.sep + "Frankfurt_LH_2303" + ".h5"
                       ]
    print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))
    queue = Queue()
    batch_size = 32
    epochs = 2000
    scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg_new.bin"
    model_file = '../trained_models/dl_model_0526'
    scaler = load_scaler(scaler_file)

    load_data(queue,epochs=1,flight_files=data_files_test,threads=6,sample_fraction=0.0005, random=True,quick_sample=False)
    X_test, y_test = queue.get()
    cols_numeric = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
    X_test_numeric = X_test[cols_numeric]
    X_test[cols_numeric] = scaler.transform(X_test_numeric)

    dl_model = SequentialModel(
        build_new=True,
        model_path=model_file,
        params = {
            "lr": 0.0001,
            "input_dims": (X_test.shape[1],),
            "output_dims": 1,
            "layer_sizes": (1024, 512, 256),
            "dropout_rate": 0.2,
            "activation": "relu",
            "loss": "MAE",
        }
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=25,
        verbose=1,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7)

    load_data(queue, epochs=1, flight_files=data_files, threads=6, sample_fraction=0.005, random=False, quick_sample=False)
    X, y = queue.get()

    # scale numeric features
    cols_numeric = ["distance", "altitude", "geoaltitude", "vertical_rate", "groundspeed"]
    X_numeric = X[cols_numeric]
    X[cols_numeric] = scaler.transform(X_numeric)

    print("loading data finished. Fitting on new batch")

    gen = batch_generator(
        df=X,
        y=y,
        batchsize=batch_size,
    )

    dl_model.model.fit(
        gen,
        max_queue_size=2000,
        validation_data=(X_test,y_test),
        callbacks=[early_stopping,reduce_lr],
        epochs=epochs,
        steps_per_epoch=len(X) // batch_size,
    )

    dl_model.model.evaluate(X_test,y_test)
    dl_model.save_model()
