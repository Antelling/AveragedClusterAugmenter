from sklearn.datasets import load_boston as load
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from numpy import mean
from ACA import ACATransformer
from sklearn import cluster
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(*load(return_X_y=True))

def score(estimator, new_X, new_y):
    estimator.fit(new_X, new_y)
    predictions = estimator.predict(X_test)
    error = mean((predictions - y_test) ** 2)
    return -error

pipeline = Pipeline([
    ("estimator", SVR())
])

params = {
    "estimator__C": [.5, 1, 2, 5, 10],
    "estimator__epsilon": [.5, 1, 1.5, 2, 2.5, 3],
    "estimator__kernel": ["rbf"],
}

print(score(RandomForestRegressor(), X_train, y_train))

#get accuracy with no preprocessing step
grid_cv = GridSearchCV(clone(pipeline), param_grid=params, scoring=score)
grid_cv.fit(X_train, y_train)
print(grid_cv.best_params_)
print(grid_cv.best_score_)

#test different preprocessing parameters
for percentage in [.6, .7, .8, .9]:
    for clusterer in [cluster.SpectralClustering(), cluster.KMeans(), cluster.AgglomerativeClustering(), cluster.Birch()]:
        print("--------------------------------------" + str(percentage * 100) + "% " + str(clusterer))
        simplifier = ACATransformer(percentage=percentage, clusterer=clusterer)
        new_X, new_y = simplifier.fit_transform(X_train, y_train)

        grid_cv = GridSearchCV(clone(pipeline), param_grid=params, scoring=score)
        grid_cv.fit(new_X, new_y)
        print(grid_cv.best_params_)
        print(grid_cv.best_score_)