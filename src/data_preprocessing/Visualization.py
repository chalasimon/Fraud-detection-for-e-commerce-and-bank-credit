# import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, data: pd.DataFrame):
        # initialize the data with the provided DataFrame
        self.data = data

    def plot_missing_values(self):
        # plot missing values in the DataFrame
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()
    # univariate analysis of the columns
    def univariate_analysis(self):
        num_cols = self.data.select_dtypes(include=['number']).columns

        # Numerical columns
        print(50 * "*")
        print("Numerical columns for visualization:")
        print(50 * "-")
        if len(num_cols) > 0:
            fig, axes = plt.subplots(nrows=len(num_cols), ncols=2, figsize=(12, 5 * len(num_cols)))
            if len(num_cols) == 1:
                axes = [axes]  # make it iterable
            
            for i, col in enumerate(num_cols):
                sns.histplot(self.data[col], kde=True, ax=axes[i][0])
                axes[i][0].set_title(f'Distribution of {col}')
                axes[i][0].set_xlabel(col)
                axes[i][0].set_ylabel('Frequency')

                sns.boxplot(x=self.data[col], ax=axes[i][1], orient='h')
                axes[i][1].set_title(f'Boxplot of {col}')
                axes[i][1].set_xlabel(col)

            plt.tight_layout()
            plt.show()
        # specify some categorical columns for visualization
        print(50 * "*")
        print("Categorical columns for visualization:")
        print(50 * "-")
        cat_cols = ['sex','browser','class']  

        # Categorical columns
        if len(cat_cols) > 0:
            fig, axes = plt.subplots(nrows=len(cat_cols), ncols=2, figsize=(12, 5 * len(cat_cols)))
            if len(cat_cols) == 1:
                axes = [axes]  # make it iterable

            for i, col in enumerate(cat_cols):
                sns.countplot(data=self.data, x=col, ax=axes[i][0])
                axes[i][0].set_title(f'Count Plot of {col}')
                axes[i][0].set_xlabel(col)
                axes[i][0].set_ylabel('Count')
                axes[i][0].tick_params(axis='x', rotation=45)

                # Pie chart
                self.data[col].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[i][1])
                axes[i][1].set_ylabel('')
                axes[i][1].set_title(f'Pie Chart of {col}')

            plt.tight_layout()
            plt.show()
        # freqency analysis of signup_time and purchase_time
        print(50 * "*")
        print("Frequency analysis of signup_time and purchase_time: by month and hour")
        print(50 * "-")
        if 'signup_time' in self.data.columns:
            self.data['signup_month'] = self.data['signup_time'].dt.month
            signup_counts = self.data['signup_month'].value_counts().sort_index()
            signup_counts.plot(kind='bar', figsize=(12, 6), title='Signups by Month')
            plt.xlabel('Month')
            plt.ylabel('Number of Signups')
            plt.xticks(rotation=45)
            plt.show()
        if 'purchase_time' in self.data.columns:
            self.data['purchase_month'] = self.data['purchase_time'].dt.month
            purchase_counts = self.data['purchase_month'].value_counts().sort_index()
            purchase_counts.plot(kind='bar', figsize=(12, 6), title='Purchases by Month')
            plt.xlabel('Month')
            plt.ylabel('Number of Purchases')
            plt.xticks(rotation=45)
            plt.show()
    def bivariate_analysis(self):
        # Bivariate analysis of numerical columns
        num_cols = self.data.select_dtypes(include=['number']).columns
        if len(num_cols) > 1:
            sns.pairplot(self.data[num_cols])
            plt.suptitle('Pairplot of Numerical Features', y=1.02)
            plt.show()
    # geolocation_analysis
    def geolocation_analysis(self, df, column):
        # filter top 10
        top_10 = df[column].value_counts().nlargest(10)
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=column, order=top_10.index)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    def class_distribution(self, df, target_column):
        # plot class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=target_column)
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    def class_distribution_over_sex(self):
        # Ensure class is categorical and ordered
        self.data['class'] = self.data['class'].astype(int)

        # Count occurrences
        class_counts = self.data.groupby(['sex', 'class']).size().reset_index(name='count')

        # Sort for consistent color order (0: Not Fraud, 1: Fraud)
        class_counts['class'] = class_counts['class'].map({0: 'Not Fraud (0)', 1: 'Fraud (1)'})

        # Define color palette matching the order of class labels
        palette = {'Not Fraud (0)': '#91d5ff', 'Fraud (1)': '#ff7875'}

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=class_counts,
            x='sex',
            y='count',
            hue='class',
            palette=palette
        )

        plt.title('Transaction Class Distribution by Sex', fontsize=14)
        plt.xlabel('Sex')
        plt.ylabel('Transaction Count')
        plt.legend(title='Class')
        plt.tight_layout()
        plt.show()