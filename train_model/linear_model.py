from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
class LinearModel:
    def __init__(self, cols, pol_degree = 1, scaler = None, pol_only = True):
        self.feature_columns = cols
        self.pol_degree = pol_degree
        self.scaler = scaler
        self.model = LinearRegression()
        self.pol_only = pol_only

    def preprocess(self, features, features_to_scale):
        if self.scaler == None:
            self.scaler = StandardScaler()
            self.scaler.fit(features[features_to_scale])
        X = features.copy()
        X[features_to_scale] = self.scaler.transform(X[features_to_scale])

        if self.pol_degree > 1:
            poly = PolynomialFeatures(self.pol_degree, interaction_only=True, include_bias=False)

            if self.pol_only:
                X = np.hstack([X[self.feature_columns]**(i+1) for i in range(self.pol_degree)])

            else:
                X = poly.fit_transform(X[self.feature_columns])

        else:
            X = X[self.feature_columns]

        return X

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mae, r2





