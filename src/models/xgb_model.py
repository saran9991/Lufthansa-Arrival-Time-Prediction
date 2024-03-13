from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from joblib import load, dump
import pandas as pd

class XGBModel:
    def __init__(self, model_file=None, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs) if model_file is None else load(model_file)

    def feature_importance_(self):
        return self.model.feature_importances_

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> (float, float):
        y_pred = self.predict(X_test)
        mae_relative = mean_absolute_error(y_test, y_pred/ X_test['distance'])
        mae = mean_absolute_error(y_test * X_test['distance'], y_pred)
        r2 = r2_score(y_test * X_test['distance'], y_pred)
        r2_relative = r2_score(y_test, y_pred/X_test['distance'])
        return mae,mae_relative, r2, r2_relative

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(self.model.predict(X))
        predictions *= X['distance']
        return predictions

    def save_model(self, file_name: str) -> None:
        dump(self.model, file_name)
