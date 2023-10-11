from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import load

import numpy as np
from src.processing_utils.preprocessing import generate_aux_columns


class LinearModel:
    def __init__(self, features, pol_degree=1, model_file=None, scaler=None, pol_only=True, cols_to_scale=None):
        self.feature_columns = features
        self.pol_degree = pol_degree
        self.scaler = scaler
        self.model = LinearRegression() if model_file is None else load(model_file)
        self.pol_only = pol_only
        self.cols_to_scale = cols_to_scale

    def preprocess(self, features, features_to_scale=None):
        if features_to_scale is None:
            features_to_scale = self.cols_to_scale
        if self.scaler == None:
            self.scaler = StandardScaler()
            self.scaler.fit(features[features_to_scale])
        X = features.copy()
        X = generate_aux_columns(X)
        X[features_to_scale] = self.scaler.transform(X[features_to_scale])
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
            X_pred = self.preprocess(X, self.cols_to_scale)
            return self.model.predict(X_pred)
        else:
            return self.model.predict(X)
