from matplotlib import pyplot as plt

import numpy as np

X = [
    1,
    2,
    2,
    3,
    5,
    6,
    9,
    10,
    12,
    13,
    15,
    15,
    17,
    19
]

y = [
    4,
    -3,
    6,
    0,
    2,
    7,
    0,
    6,
    1,
    9,
    4,
    12,
    7,
    12
]

#plot base points
plt.scatter(X, y, color="red")

#ready a svr on base points
from sklearn.svm import SVR
r = SVR(C=5)
sklearn_x = [[x] for x in X]
r.fit(sklearn_x, y)

#plot predictions of base svr
points_to_graph = [[x] for x in np.linspace(start=0, stop=20, num=200)]
predictions = r.predict(points_to_graph)
plt.plot(points_to_graph, predictions, label="Base SVR")

from ACA import ACAWrapper
from sklearn.cluster import KMeans
aca = ACAWrapper(r, percentage=.5, return_old=False)
aca.fit(sklearn_x, y)

predictions = aca.predict(points_to_graph)
plt.plot(points_to_graph, predictions, label="Augmented SVR")

plt.legend()
plt.show()