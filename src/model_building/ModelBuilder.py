# import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

class ModelBuilder:
    def __init__(self, data: pd.DataFrame):
        # initialize the data with the provided DataFrame
        self.data = data

    def prepare_data(self):
        # Convert 'purchase_time' to datetime
        if 'purchase_time' in self.data.columns:
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
            # Drop raw datetime and redundant columns
            self.data.drop(columns=['purchase_time'], inplace=True)
        if 'signup_time' in self.data.columns:
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
            # Drop raw datetime and redundant columns
            self.data.drop(columns=['signup_time'], inplace=True)
        # Handle categorical string columns
        object_cols = self.data.select_dtypes(include='object').columns

        for col in object_cols:
            if col == 'device_id':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
            else:
                self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
        # Clean column names to remove special JSON characters
        self.data.columns = self.data.columns.str.replace(r'[{}[\]":,]', '', regex=True)
        self.data.columns = self.data.columns.str.replace(r'\W+', '_', regex=True)
        return self.data

    def split_data(self, target, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[target])
        y = self.data[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def handle_imbalance(self, X_train, y_train):
        # handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Logistic Regression training time: {end_time - start_time:.2f} seconds")
        return model

    def train_gradient_boosting(self, X_train, y_train):
        model = LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Gradient Boosting training time: {end_time - start_time:.2f} seconds")
        return model

    def evaluate_model(self, model, X_test, y_test):
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
    def save_model(self, model, filename):
        # Save the model to a file using pickle
        pickle_path = '../models/'
        with open(pickle_path + filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {pickle_path + filename}")

    def run_pipeline(self, target='class', test_size=0.2, random_state=42,data=None):
        # Prepare data
        print("Preparing data...")
        self.prepare_data()

        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(target, test_size, random_state)
        
        # Handle class imbalance
        print("Handling class imbalance...")
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Train models
        print("Training Logistic Regression model...")
        lr_model = self.train_logistic_regression(X_train_resampled, y_train_resampled)
        print("Training Gradient Boosting model...")
        gb_model = self.train_gradient_boosting(X_train_resampled, y_train_resampled)
        
        # Evaluate models
        print("Evaluating models...")
        print("Logistic Regression Model Evaluation:")
        self.evaluate_model(lr_model, X_test, y_test)
        print("\nGradient Boosting Model Evaluation:")
        self.evaluate_model(gb_model, X_test, y_test)
        # Save models
        print("Saving models...")
        if data == "fraud":
            self.save_model(lr_model, 'fraud_logistic_regression_model.pkl')
            self.save_model(gb_model, 'fraud_gradient_boosting_model.pkl')
        elif data == "credit":
            self.save_model(lr_model, 'credit_logistic_regression_model.pkl')
            self.save_model(gb_model, 'credit_gradient_boosting_model.pkl')
