# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        # initialize the data with the provided DataFrame
        self.data = data

    def add_time_based_features(self):
        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek
        self.data['time_since_signup'] = (self.data['purchase_time'] - self.data['signup_time']).dt.total_seconds() / 3600
        return self.data

    def add_transaction_frequency_and_velocity(self):
        # Frequency: count of transactions per ip address
        self.data['transaction_frequency']= self.data.groupby('ip_address')['ip_address'].transform('count')
        self.data = self.data.sort_values(by=['ip_address', 'purchase_time'])
        # Time difference in seconds between consecutive transactions from same IP
        self.data['time_diff'] = self.data.groupby('ip_address')['purchase_time'].diff().dt.total_seconds().fillna(0)
        return self.data