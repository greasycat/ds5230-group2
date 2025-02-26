from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt

# define the parameter grid


def silhouette_score_spectral_metric(data, fit):
    return silhouette_score(data, fit.row_labels_)

def davies_bouldin_score_spectral_metric(data, fit):
    return davies_bouldin_score(data, fit.row_labels_)

def calinski_harabasz_score_spectral_metric(data, fit):
    return calinski_harabasz_score(data, fit.row_labels_)


def hyperparam_search(data, model, param_grid, metrics_dict, label_getter):
    scores = {}

    for params in product(*param_grid.values()):
        model.set_params(**dict(zip(param_grid.keys(), params)))
        fit = model.fit(data)
        for metric_name, metric in metrics_dict.items():
            score = metric(data, fit)
            if metric_name not in scores:
                scores[metric_name] = []
            scores[metric_name].append((score, label_getter(fit), params))
            print(f"Score: {score}, Params: {params}, Metric: {metric_name}")

    # sort the scores
    for metric_name in scores:
        scores[metric_name].sort(key=lambda x: x[0], reverse=True)
    return scores

def spectral_clustering(data, param_grid, random_seed=42):
    model = SpectralCoclustering(random_state=random_seed)
    metrics_dict = {
        "silhouette_score": silhouette_score_spectral_metric,
        "davies_bouldin_score": davies_bouldin_score_spectral_metric,
        "calinski_harabasz_score": calinski_harabasz_score_spectral_metric
    }
    return hyperparam_search(data, model, param_grid, metrics_dict, lambda fit: (fit.row_labels_, fit.column_labels_))

def plot_clustering_on_pca_spectral(data, fit, name):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=fit.row_labels_)
    ax.set_title("Clustering on PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    fig.savefig(f"figs/spectral_clustering_on_pca_{name}.png")

