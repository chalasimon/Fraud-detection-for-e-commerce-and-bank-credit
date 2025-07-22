# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        df = self.data
        # Frequency: count of transactions per IP address
        ip_freq = df.groupby('ip_address')['purchase_time'].count().rename("transaction_count")
        df = df.merge(ip_freq, on='ip_address', how='left')
        # Velocity: time between transactions
        df = df.sort_values(by=['ip_address', 'purchase_time'])
        df['prev_time'] = df.groupby('ip_address')['purchase_time'].shift(1)
        df['velocity'] = (df['purchase_time'] - df['prev_time']).dt.total_seconds() / 60  # in minutes
        df['velocity'].fillna(df['velocity'].median(), inplace=True)
        return df

