from tensorflow import config as tf_config
from joblib import load as load_joblib
from fit_model import fit_model
from sequential_model import SequentialModel
import os


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))
    scaler_file = ".." + os.sep + "trained_models" + os.sep + "std_scaler_reg_new.bin"
    model_file = ".." + os.sep + "trained_models" + os.sep + "dl_model_0602_no_coord"
    scaler = load_joblib(scaler_file)

    dl_model = SequentialModel(
        build_new=True,
        model_path=model_file,
        params={
            "lr": 0.0001,
            "input_dims": (19,),
            "output_dims": 1,
            "layer_sizes": (1024, 512, 256),
            "dropout_rate": 0.2,
            "activation": "relu",
            "loss": "MAE",
        }
    )
    fit_model(
        dl_model.model,
        scaler,
        with_weights=False,
        load_new_test_data=False,
        load_new_training_data=False,
        fraction_train=1,
        fraction_test=1,
        batch_size=32,
        drop_columns=["arrival_time", "timestamp", "track", "latitude", "longitude",  "bearing_sin", "bearing_cos", "latitude_rad", "longitude_rad"]
    )
    dl_model.save_model()
