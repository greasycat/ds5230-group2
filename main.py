import src.prep as prep
import joblib

try:
    prep = joblib.load("prep.pkl")
except FileNotFoundError:
    prep = prep.Prep(output_pca=False)
    joblib.dump(prep, "prep.pkl")

print(prep.prep_car.head())

from src.clustering import spectral_clustering

try:
    scores = joblib.load("spectral_clustering.pkl")
except FileNotFoundError:
    scores = spectral_clustering(prep.prep_car)
    joblib.dump(scores, "spectral_clustering.pkl")


print(scores[0][2:])
