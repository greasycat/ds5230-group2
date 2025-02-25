import src.prep as prep
from src.clustering import *


prep = prep.Prep(output_pca=True, render=False)


param_grid_spectral_car = {
    'n_clusters': [5,6,7],
}

param_grid_spectral_customer = {
    'n_clusters': [2,3,4,5],
}

print(prep.prep_car.head())

# clustering the car dataset
spectral_scores_car = spectral_clustering(prep.prep_car, param_grid_spectral_car)

# clustering the customer dataset
spectral_scores_customer = spectral_clustering(prep.prep_customer, param_grid_spectral_customer)
