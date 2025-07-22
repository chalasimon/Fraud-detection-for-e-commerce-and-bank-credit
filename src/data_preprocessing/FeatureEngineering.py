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


