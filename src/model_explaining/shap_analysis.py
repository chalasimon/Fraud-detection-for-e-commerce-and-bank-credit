import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle

class ShapAnalysis:
    def __init__(self, model, X_train, y_train=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        # Initialize SHAP explainer
        if hasattr(model, "predict_proba"):
            self.explainer = shap.Explainer(model.predict, X_train)
        else:
            self.explainer = shap.Explainer(model, X_train)

        print("SHAP Explainer initialized.")

    def compute_shap_values(self):
        print("Computing SHAP values...")
        self.shap_values = self.explainer(self.X_train)
        print("SHAP values computed.")

    def summary_plot(self, save_path=None):
        print("Generating SHAP summary plot...")
        shap.summary_plot(self.shap_values, self.X_train, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"SHAP summary plot saved to {save_path}")
        plt.show()

    def force_plot(self, row_index=0, save_path=None):
        print(f"Generating SHAP force plot for index {row_index}...")
        force = shap.plots.force(
            self.explainer.expected_value[0],
            self.shap_values[row_index].values,
            self.X_train.iloc[row_index],
            matplotlib=True,
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"SHAP force plot saved to {save_path}")
        plt.show()
def run_analysis(self, data_name=None):
    # Load models
    models = {
        "fraud_logistic": pickle.load(open("../models/fraud_logistic_regression_model.pkl", "rb")),
        "fraud_gb": pickle.load(open("../models/fraud_gradient_boosting_model.pkl", "rb")),
        "credit_logistic": pickle.load(open("../models/credit_logistic_regression_model.pkl", "rb")),
        "credit_gb": pickle.load(open("../models/credit_gradient_boosting_model.pkl", "rb"))
    }

    # Load data
    data = {
        "fraud": pd.read_csv("../data/processed/resampled_X_train_fraud.csv"),
        "credit": pd.read_csv("../data/processed/resampled_X_train_credit.csv")
    }
    # Run check for data_name
    if data_name == "fraud":
        X_train = data["fraud"]
        model = models["fraud_gb"]
    elif data_name == "credit":
        X_train = data["credit"]
        model = models["credit_gb"]
    else:
        raise ValueError("Invalid data_name. Choose 'fraud' or 'credit'.")
    # run SHAP analysis
    shap_analysis = ShapAnalysis(model, X_train)
    shap_analysis.compute_shap_values()
    shap_analysis.summary_plot(save_path=f"shap_summary_plot_{data_name}.png")
    shap_analysis.force_plot(row_index=0, save_path=f"../screenshots/shap_force_plot_{data_name}.png")
    print("SHAP analysis completed.")
