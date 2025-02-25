from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import silhouette_score
from itertools import product
import matplotlib.pyplot as plt

# define the parameter grid


def silhouette_score_spectral_metric(data, fit):
    return silhouette_score(data, fit.row_labels_)


def hyperparam_search(data, model, param_grid, metric):
    scores = []

    for params in product(*param_grid.values()):
        model.set_params(**dict(zip(param_grid.keys(), params)))
        model.fit(data)
        score = metric(data, model)
        scores.append((score, model, params))
        print(f"Score: {score}, Params: {params}")

    # sort the scores
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores

def spectral_clustering(data, param_grid, random_seed=42):
    model = SpectralCoclustering(random_state=random_seed)
    metric = silhouette_score_spectral_metric
    return hyperparam_search(data, model, param_grid, metric)

def plot_clustering_on_pca_spectral(data, fit, name):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=fit.row_labels_)
    ax.set_title("Clustering on PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    fig.savefig(f"figs/spectral_clustering_on_pca_{name}.png")

