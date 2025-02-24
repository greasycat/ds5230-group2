import src.prep as prep
from src.clustering import spectral_clustering
import joblib

import sys

# Check if --cached parameter is provided in argv
is_cached = '--cached' in sys.argv


prep = prep.Prep(output_pca=True)

param_grid_spectral_car = {
    'n_clusters': [5,6,7],
    'affinity': ['rbf'],
    'gamma': [0.1]
}

param_grid_spectral_customer = {
    'n_clusters': [2,3,4,5],
    'affinity': ['rbf'],
    'gamma': [0.1]
}

def load_or_compute_scores(file_name, data, param_grid):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        scores = spectral_clustering(data, param_grid)
        joblib.dump(scores, file_name)
        return scores

scores_car = load_or_compute_scores("spectral_scores_car.pkl", prep.prep_car, param_grid_spectral_car) if not is_cached else spectral_clustering(prep.prep_car, param_grid_spectral_car)
scores_customer = load_or_compute_scores("spectral_scores_customer.pkl", prep.prep_customer, param_grid_spectral_customer) if not is_cached else spectral_clustering(prep.prep_customer, param_grid_spectral_customer)

