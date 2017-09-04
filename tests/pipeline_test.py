from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_linnerud, load_iris
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from ACA import ACAWrapper
from sklearn import cluster
from copy import copy


def score(estimator, new_X, new_y):
    estimator.fit(new_X, new_y)
    predictions = estimator.predict(X)
    error = np.mean((predictions - y) ** 2)
    return -error


augmented_regressor = ACAWrapper(estimator=SVR())
base_regressor = SVR()

base_params = {
    "C": [.01, .1, 1],
    "epsilon": [.001, .1, .5],
    "kernel": ["rbf"],
}
augmented_params = {
    "estimator__C": [.01, .1, 1],
    "estimator__epsilon": [.001, .1, .5],
    "estimator__kernel": ["rbf"],
    "percentage": [.1, .2, .5],
    "clusterer": [cluster.KMeans()]
}

base_cv = GridSearchCV(base_regressor, param_grid=base_params, scoring=score)
augmented_cv = GridSearchCV(augmented_regressor, param_grid=augmented_params, scoring=score)

for l in load_boston, load_diabetes:
    print("current dataset: " + l.__name__)
    X, y = l(return_X_y=True)

    for cv in base_cv, augmented_cv:
        cv.fit(X, y)
        print(cv.best_params_)
        print(cv.best_score_)
