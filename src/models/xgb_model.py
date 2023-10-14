from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from joblib import load, dump
import pandas as pd
from src.processing_utils.preprocessing import generate_aux_columns
from src.processing_utils.h3_preprocessing import get_h3_index, add_density

class XGBModel:
    def __init__(self, features, model_file=None, **kwargs):
        self.feature_columns = features
        self.model = xgb.XGBRegressor(**kwargs) if model_file is None else load(model_file)

    def feature_importance_(self):
        return self.model.feature_importances_

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data[self.feature_columns]
        data = generate_aux_columns(data)
        data = get_h3_index(data, 4)
        data = add_density(data)
        return data

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, preprocess=False) -> None:
        if(preprocess):
            X_train = self.preprocess(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, preprocess = False) -> (float, float):
        if(preprocess):
            X_test = self.preprocess(X_test)
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, r2

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X))

    def save_model(self, file_name: str) -> None:
        dump(self.model, file_name)
