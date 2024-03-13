from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import load

import numpy as np
from src.processing_utils.preprocessing import generate_aux_columns


class LinearModel:
    def __init__(
            self,
            features,
            pol_degree=1,
            model_file=None,
            std_scaler=None,
            minmax_scaler=None,
            pol_only=True,
            cols_to_scale_std=None,
            cols_to_scale_minmax=None
    ):
        self.feature_columns = features
        self.pol_degree = pol_degree
        self.std_scaler = std_scaler
        self.minmax_scaler = minmax_scaler
        self.model = LinearRegression() if model_file is None else load(model_file)
        self.pol_only = pol_only
        self.cols_to_scale_std = cols_to_scale_std
        self.cols_to_scale_minmax = cols_to_scale_minmax

    def preprocess(self, features, features_to_scale_std=None, features_to_scale_minmax=None):
        if features_to_scale_std is None:
            features_to_scale_std = self.cols_to_scale_std
        if features_to_scale_minmax is None:
            features_to_scale_minmax = self.cols_to_scale_minmax
        X = features.copy()
        X = generate_aux_columns(X)

        if self.std_scaler == None:
            self.std_scaler = StandardScaler()
            self.std_scaler.fit(X[features_to_scale_std])
        if self.minmax_scaler == None:
            self.minmax_scaler = MinMaxScaler()
            self.minmax_scaler.fit(X[features_to_scale_minmax])
        X[features_to_scale_std] = self.std_scaler.transform(X[features_to_scale_std])
        X[features_to_scale_minmax] = self.minmax_scaler.transform(X[features_to_scale_minmax])

        if self.pol_degree > 1:
            poly = PolynomialFeatures(self.pol_degree, interaction_only=True, include_bias=False)

            if self.pol_only:
                X = np.hstack([X[self.feature_columns] ** (i + 1) for i in range(self.pol_degree)])

            else:
                X = poly.fit_transform(X[self.feature_columns])

        else:
            X = X[self.feature_columns]

        return X

    def fit(self, X_train, y_train):
        X = self.preprocess(X_train)
        self.model.fit(X, y_train)

    def evaluate(self, X_test, y_test, preprocess=True):
        y_pred = self.predict(X_test, preprocess)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mae, r2

    def predict(self, X, preprocess=True):
        if preprocess:
            X_pred = self.preprocess(
                X,
                features_to_scale_std=self.cols_to_scale_std,
                features_to_scale_minmax=self.cols_to_scale_minmax,
            )
            return self.model.predict(X_pred)
        else:
            return self.model.predict(X)
