import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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
