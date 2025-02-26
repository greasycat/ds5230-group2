import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

CAR_NUMERIC_COLUMNS =[
    'highway08',
    'city08',
    'displ',
    'year',
    'cylinders',
]

CAR_CATEGORICAL_COLUMNS = [
    'fueltype',
    'drive',
    'trany',
    'vclass',
]


CUSTOMER_NUMERIC_COLUMNS = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

CUSTOMER_CATEGORICAL_COLUMNS = [
    'Gender'
]

def plot_pca(pca_result, fig_suffix, output=False, render=False):
    # check if less than 3 components
    if pca_result.shape[1] < 3:
        print("PCA result has less than 3 components, skipping plot")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[0], pca_result[1], c=pca_result[2])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of {fig_suffix} Data')
    if render:
        print(f"Rendering PCA plot for {fig_suffix} data")
        plt.show()
    if output:
        print(f"Saving PCA plot for {fig_suffix} data")
        plt.savefig(f'figs/pca_plot_{fig_suffix}.png')

def plot_pca_contribution(pca, features, fig_suffix, output=False, render=False):
    components_df = pd.DataFrame(pca.components_, columns=features, index=[f'PC{i+1}' for i in range(pca.n_components_)])

    plt.figure(figsize=(12, 8))
    x = np.arange(len(components_df.columns))  # the label locations
    width = 0.15  # the width of the bars

    for i in range(components_df.shape[0]):
        bars = plt.bar(x + i * width, components_df.iloc[i], width, label=f'PC{i+1}', alpha=0.5)
        # Annotate each bar with its contribution value
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xlabel('Features')
    plt.ylabel('Absolute Contribution')
    plt.title('Feature Contributions to Principal Components')
    plt.xticks(x + width * (components_df.shape[0] - 1) / 2, components_df.columns, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if render:
        print(f"Rendering PCA contribution plot for {fig_suffix} data")
        plt.show()
    if output:
        print(f"Saving PCA contribution plot for {fig_suffix} data")
        plt.savefig(f'figs/pca_contribution_plot_{fig_suffix}.png')

def plot_explained_variance(cumulative_variance_ratio, explained_variance_ratio, n_components_95, fig_suffix, output=False, render=False):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components')
    plt.legend()
    plt.grid(True)
    if render:
        print(f"Rendering PCA variance plot for {fig_suffix} data")
        plt.show()
    if output:
        print(f"Saving PCA variance plot for {fig_suffix} data")
        plt.savefig(f'figs/pca_variance_plot_{fig_suffix}.png')

class Prep:
    def __init__(self, car_data_path=None, customer_data_path=None, random_seed=42, output_pca=False, render=False):
        self.car_data_path = "data/car.csv" if car_data_path is None else car_data_path
        self.customer_data_path = "data/Mall_Customers.csv" if customer_data_path is None else customer_data_path

        # set random seed
        self.random_seed = random_seed
        
        self.output_pca = output_pca
        self.render = render

        self._prep_car_data()
        self._prep_customer_data()
    
    
    def _prep_car_data(self):
        self.raw_car = pd.read_csv(self.car_data_path, sep=";")
        self._clean_car_data()
        pca_result = self._pca(self.car[CAR_NUMERIC_COLUMNS], "car", CAR_NUMERIC_COLUMNS)
        one_hot_encoded = self._one_hot_encode(self.car[CAR_CATEGORICAL_COLUMNS])
        self.prep_car = pd.concat([pca_result, one_hot_encoded], axis=1)

    def _clean_car_data(self):
        self.car = self.raw_car.sample(frac=0.2, random_state=self.random_seed)

        # fill NAs with "EV or Others" for categorical columns
        for column in CAR_CATEGORICAL_COLUMNS:
            self.car[column] = self.car[column].fillna("EV or Others")
        
        for column in CAR_NUMERIC_COLUMNS:
            self.car[column] = self.car[column].fillna(0)

        # convert cylinders to str
        # self.car['cylinders'] = self.car['cylinders'].astype(str)

    def _pca(self, data, fig_suffix, features):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Initialize PCA without specifying n_components to keep all components initially
        pca = PCA()
        # Fit PCA
        pca_result = pca.fit_transform(scaled)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Find number of components needed to explain 95% of variance
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

        # Plot explained variance ratio
        if self.output_pca or self.render:
            plot_explained_variance(cumulative_variance_ratio, explained_variance_ratio, n_components_95, fig_suffix, output=self.output_pca, render=self.render)

        # Refit PCA with optimal number of components
        pca = PCA(n_components=n_components_95)
        pca_result = pd.DataFrame(pca.fit_transform(scaled))

        if self.output_pca or self.render:
            plot_pca_contribution(pca, features, fig_suffix, output=self.output_pca, render=self.render)
            plot_pca(pca_result, fig_suffix, output=self.output_pca, render=self.render)

        return pca_result
    
    def _one_hot_encode(self, data):
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(data)
        one_hot_encoded = pd.DataFrame(one_hot_encoded)
        return one_hot_encoded

    def _prep_customer_data(self):
        self.raw_customer = pd.read_csv(self.customer_data_path)

        pca_result = self._pca(self.raw_customer[CUSTOMER_NUMERIC_COLUMNS], "customer", CUSTOMER_NUMERIC_COLUMNS)
        one_hot_encoded = self._one_hot_encode(self.raw_customer[CUSTOMER_CATEGORICAL_COLUMNS])
        self.prep_customer = pd.concat([pca_result, one_hot_encoded], axis=1)


