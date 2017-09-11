from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.base import clone
import numpy as np
import warnings


def _average_clusters(clusterer, X, Y, percentage, return_old):
    if percentage is not None:
        try:
            clusterer.set_params(n_clusters=int(len(X) * percentage))
        except ValueError:
            warnings.warn(
                "Clusterer passed to DataSimplifier has not attribute n_clusters, percentage parameter has no effect")

    clusterer = clusterer.fit(X)
    groups = {}
    for i, label in enumerate(clusterer.labels_):
        if not label in groups:
            groups[label] = []
        groups[label].append((X[i], Y[i]) if Y is not None else X[1])

    if Y is None:
        # now we want to find the average of each group
        for group in groups:
            groups[group] = np.mean(groups[group], axis=0).tolist()

        new_x = np.array(list(groups.values()))
        if return_old:
            return np.concatenate([X, new_x])
        return new_x

    else:
        new_x = []
        new_y = []
        for _, group in groups.items():
            x = [g[0] for g in group]
            new_x.append(np.mean(x, axis=0).tolist())

            y = [g[1] for g in group]
            new_y.append(np.mean(y, axis=0).tolist())

        if return_old:
            return np.concatenate([X, new_x]), np.concatenate([Y, new_y])
        return (np.array(new_x), np.array(new_y))


class ACATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clusterer=None, percentage=None, return_old=True):
        self.clusterer = clusterer
        self.percentage = percentage
        self.return_old = return_old

    def fit(self, X, y=None):  # fitting this transformer is meaningless
        return self

    def transform(self, X, y=None):  # short-circuit to fit_transform
        return self.fit_transform(X, y)

    def fit_transform(self, X, Y=None, **fit_params):
        if not (type(X) is np.ndarray or type(X) is list):
            raise TypeError("ACA does not accept sparse input")

        if self.clusterer is not None:
            clusterer = clone(self.clusterer)
        else:
            clusterer = KMeans()
        return _average_clusters(clusterer, X, Y, self.percentage, self.return_old)


class ACAWrapper(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator=None, clusterer=None, percentage=None, return_old=True, disabled=False):
        self.clusterer = clusterer
        self.percentage = percentage
        self.return_old = return_old
        self.base_estimator = estimator
        self.disabled = disabled # easy way to disable feature for pipelines and param searches

    def fit(self, X, y):
        if not (type(X) is np.ndarray or type(X) is list):
            raise TypeError("ACA does not accept sparse input")
        if self.base_estimator is not None:
            estimator = clone(self.base_estimator)
        else:
            estimator = SVR()

        if self.clusterer is not None:
            clusterer = clone(self.clusterer)
        else:
            clusterer = KMeans()


        if not self.disabled:
            X, y = _average_clusters(clusterer, X, y, self.percentage, self.return_old)

        estimator.fit(X, y)
        self.estimator_ = estimator

        return self

    def predict(self, X):
        return self.estimator_.predict(X)