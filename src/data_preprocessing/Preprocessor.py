# import the libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        # initializa the data with the provided DataFrame
        self.data = data

    def identify_missing_values(self):
        # identify columns with missing values
        missing_values = self.data.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        return missing_columns
    def handle_missing_values(self):
        # handle missing values by dropping rows with any missing values
        self.data.dropna(inplace=True)
        return self.data
    def remove_duplicates(self):
        # remove duplicate rows from the DataFrame
        self.data.drop_duplicates(inplace=True)
        return self.data
    def correct_data_types(self):
        # date time columns are converted to datetime type
        # convert signup_time and purchase_time to datetime
        if 'signup_time' in self.data.columns:
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'], errors='coerce')
        if 'purchase_time' in self.data.columns:
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'], errors='coerce')
        # convert class column to categorical type
        if 'class' in self.data.columns:
            self.data['class'] = self.data['class'].astype('category')
        return self.data
    def scale_numerical_features(self,df):
        # scale numerical features using StandardScaler
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df