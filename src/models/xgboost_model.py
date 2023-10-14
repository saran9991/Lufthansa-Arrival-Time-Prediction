from sklearn.metrics import mean_absolute_error, r2_score
import xgboost_model as xgb
from joblib import load, dump
import pandas as pd

class XGBModel:
    def __init__(self, features, model_file=None, **kwargs):
        self.feature_columns = features
        self.model = xgb.XGBRegressor(**kwargs) if model_file is None else load(model_file)

    def feature_importance_(self):
        return self.model.feature_importances_

    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        return features[self.feature_columns]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X = self.preprocess(X_train)
        self.model.fit(X, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, preprocess=False) -> (float, float):
        y_pred = self.predict(X_test, preprocess)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, r2

    def predict(self, X: pd.DataFrame, preprocess=True) -> pd.Series:
        if preprocess:
            X_pred = self.preprocess(X)
        else:
            X_pred = X

        return pd.Series(self.model.predict(X_pred))

    def save_model(self, file_name: str) -> None:
        dump(self.model, file_name)
