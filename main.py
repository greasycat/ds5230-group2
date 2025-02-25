import src.prep as prep
from src.clustering import *


prep = prep.Prep(output_pca=True)


param_grid_spectral_car = {
    'n_clusters': [5,6,7],
}

param_grid_spectral_customer = {
    'n_clusters': [2,3,4,5],
}

print(prep.prep_car.head())

# clustering the car dataset
spectral_scores_car = spectral_clustering(prep.prep_car, param_grid_spectral_car)
best_fit_car = spectral_scores_car[0][1]
plot_clustering_on_pca_spectral(prep.prep_car, best_fit_car, "car")

# clustering the customer dataset
spectral_scores_customer = spectral_clustering(prep.prep_customer, param_grid_spectral_customer)
best_fit_customer = spectral_scores_customer[0][1]

plot_clustering_on_pca_spectral(prep.prep_customer, best_fit_customer, "customer")
