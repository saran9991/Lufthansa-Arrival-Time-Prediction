import shap
import pandas as pd
import xgboost as xgb

def get_shap(df_train: pd.DataFrame, df_test: pd.DataFrame, model: xgb.XGBRegressor, FEATURES: list) -> None:
    """
    Compute and plot SHAP values for a given trained model and datasets.

    Parameters:
    - df_train (pd.DataFrame): The processed training dataset. Should only contain features used for model training.
    - df_test (pd.DataFrame): The processed testing dataset. Should only contain features used for model training.
    - model (trained model): The trained machine learning model. It should be already trained.
    - FEATURES (list): List of feature names.

    Note: Ensure the input datasets (df_train and df_test) are already processed and the model is trained.
    """
    explainer = shap.TreeExplainer(model, df_train)
    shap_values = explainer.shap_values(df_test)
    shap.summary_plot(shap_values, df_test, feature_names=FEATURES)
