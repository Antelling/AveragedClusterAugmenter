from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_linnerud, load_iris
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from ACA import ACATransformer
from sklearn import cluster
from sklearn.base import clone

# lets start with regression


def score(estimator, new_X, new_y):
    estimator.fit(new_X, new_y)
    predictions = estimator.predict(X)
    error = np.mean((predictions - y) ** 2)
    return -error

pipeline = Pipeline([
    ("estimator", SVR())
])

params = {
    "estimator__C": [.5, 1, 2, 5, 10],
    "estimator__epsilon": [.5, 1, 1.5, 2, 2.5, 3],
    "estimator__kernel": ["rbf"],
}

for l in load_boston, load_diabetes:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    X, y = l(return_X_y=True)

    grid_cv = GridSearchCV(clone(pipeline), param_grid=params, scoring=score)
    grid_cv.fit(X, y)
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)

    for percentage in [.6, .7, .8, .9]:
        for clusterer in [cluster.SpectralClustering(), cluster.KMeans(), cluster.AgglomerativeClustering(), cluster.Birch()]:
            print("--------------------------------------" + str(percentage * 100) + "% " + str(clusterer))
            simplifier = ACATransformer(percentage=percentage, clusterer=clusterer)
            new_X, new_y = simplifier.fit_transform(X, y)

            grid_cv = GridSearchCV(clone(pipeline), param_grid=params, scoring=score)
            grid_cv.fit(new_X, new_y)
            print(grid_cv.best_params_)
            print(grid_cv.best_score_)