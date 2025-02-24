import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

CAR_NUMERIC_COLUMNS =[
    'highway08',
    'city08',
    'displ',
    'year'
]

CAR_CATEGORICAL_COLUMNS = [
    'fueltype',
    'drive',
    'trany',
    'vclass',
    'cylinders',
]


def plot_pca(pca_result):
    # check if less than 3 components
    if pca_result.shape[1] < 3:
        print("PCA result has less than 3 components, skipping plot")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[0], pca_result[1], c=pca_result[2])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Car Data')
    plt.savefig('pca_plot.png')


class Prep:
    def __init__(self, car_data_path=None, customer_data_path=None, random_seed=42, output_pca=False):
        self.car_data_path = "data/car.csv" if car_data_path is None else car_data_path
        self.customer_data_path = "data/customer.csv" if customer_data_path is None else customer_data_path

        # set random seed
        np.random.seed(random_seed)
        
        self.output_pca = output_pca

        self._prep_car_data()

    
    def _prep_car_data(self):
        self.raw_car = pd.read_csv(self.car_data_path, sep=";")
        self._clean_car_data()
        pca_result = self._pca(self.car[CAR_NUMERIC_COLUMNS])
        one_hot_encoded = self._one_hot_encode(self.car[CAR_CATEGORICAL_COLUMNS])
        self.prep_car = pd.concat([pca_result, one_hot_encoded], axis=1)

    def _clean_car_data(self):
        self.car = self.raw_car.sample(frac=0.2)

        # fill NAs with "EV or Others" for categorical columns
        for column in CAR_CATEGORICAL_COLUMNS:
            self.car[column] = self.car[column].fillna("EV or Others")
        
        for column in CAR_NUMERIC_COLUMNS:
            self.car[column] = self.car[column].fillna(0)

        # convert cylinders to str
        self.car['cylinders'] = self.car['cylinders'].astype(str)

    def _pca(self, data):
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
        if self.output_pca:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('Explained Variance Ratio vs Number of Components')
            plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
            plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components')
            plt.legend()
            plt.grid(True)
            plt.savefig('pca_variance_plot.png')

        # Refit PCA with optimal number of components
        pca = PCA(n_components=n_components_95)
        pca_result = pca.fit_transform(scaled)
        pca_result = pd.DataFrame(pca_result)

        if self.output_pca:
            plot_pca(pca_result)

        return pca_result
    
    def _one_hot_encode(self, data):
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(data)
        one_hot_encoded = pd.DataFrame(one_hot_encoded)
        return one_hot_encoded


    def _clean_customer_data(self):
        pass
