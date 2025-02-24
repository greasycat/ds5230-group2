from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from itertools import product

# define the parameter grid


def silhouette_score_metric(data, fit):
    return silhouette_score(data, fit.labels_)


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

def spectral_clustering(data, param_grid):
    model = SpectralClustering()
    metric = silhouette_score_metric
    return hyperparam_search(data, model, param_grid, metric)
