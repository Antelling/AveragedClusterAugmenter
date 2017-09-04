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

from sklearn.svm import SVR
from ACA import ACAWrapper
from matplotlib import pyplot as plt
import numpy as np

#plot base points
plt.scatter(X, y, color="red")

#make x acceptable to sklearn.fit
sklearn_x = [[x] for x in X]

#make linspace for graphing
points_to_graph = [[x] for x in np.linspace(start=0, stop=20, num=200)]

#make svr for graphing
r = SVR(C=5)

for percentage in np.linspace(0.1, .9, 9):
    aca = ACAWrapper(r, percentage=percentage, return_old=False)
    aca.fit(sklearn_x, y)

    predictions = aca.predict(points_to_graph)
    plt.plot(points_to_graph, predictions, label="percentage: " + str(percentage))

plt.legend()
plt.show()