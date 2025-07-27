# import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreparation:
    def __init__(self, data: pd.DataFrame):
        # initialize the data with the provided DataFrame
        self.data = data

    def split_data(self, target='class', test_size=0.2, random_state=42):
        X = self.data.drop(columns=[target])
        y = self.data[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    def handle_imbalance(self, X_train, y_train):
        # handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    def train_logistic_regression(X_train, y_train):
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        return model
    def train_gradient_boosting(X_train, y_train):
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
        model.fit(X_train, y_train)
        return model
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
        
        f1 = f1_score(y_test, y_pred)
        print("F1 Score:", f1)
    def run_pipeline(self, target='class', test_size=0.2, random_state=42):
        # Split the data
        X_train, X_test, y_train, y_test = self.split_data(target, test_size, random_state)
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Train models
        lr_model = self.train_logistic_regression(X_train_resampled, y_train_resampled)
        gb_model = self.train_gradient_boosting(X_train_resampled, y_train_resampled)
        
        # Evaluate models
        print("Logistic Regression Model Evaluation:")
        self.evaluate_model(lr_model, X_test, y_test)
        
        print("\nGradient Boosting Model Evaluation:")
        self.evaluate_model(gb_model, X_test, y_test)
    