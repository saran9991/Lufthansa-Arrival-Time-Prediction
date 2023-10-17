import shap

def get_shap(df_train, df_test, model, FEATURES):
    """
    Compute and plot SHAP values for a given trained model and datasets.

    Parameters:
    - df_train (pd.DataFrame): The processed training dataset. Should only contain features used for model training.
    - df_test (pd.DataFrame): The processed testing dataset. Should only contain features used for model training.
    - model (trained model): The trained machine learning model. It should be already trained.
    - FEATURES (list): List of feature names.

    Note: Ensure the input datasets (df_train and df_test) are already processed and the model is trained.
    """
    df_train = df_train.to_numpy()
    df_test = df_test.to_numpy()

    explainer = shap.Explainer(model, df_train)
    shap_values = explainer(df_test)
    shap.summary_plot(shap_values, df_test, feature_names=FEATURES)
