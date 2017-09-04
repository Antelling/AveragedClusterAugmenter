from sklearn.datasets import load_boston as load
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from ACA import ACATransformer
from sklearn.cluster import KMeans as c
import numpy as np

X, y = load(return_X_y=True)

transformer = ACATransformer(clusterer=c(n_clusters=14), return_old=False)
averaged_x, averaged_y = transformer.fit_transform(X, y)

all_x = np.concatenate([X, averaged_x])

reduced = TSNE(n_components=2).fit_transform(all_x)

unclustered_x = reduced[0:len(X)]
clustered_x = reduced[len(X):]

print(len(X), len(unclustered_x))
print(len(averaged_x), len(clustered_x))

plot_x = [point[0] for point in unclustered_x]
plot_y = [point[1] for point in unclustered_x]
plt.scatter(plot_x, plot_y, c=y, cmap="autumn")

plot_x = [point[0] for point in clustered_x]
plot_y = [point[1] for point in clustered_x]
plt.scatter(plot_x, plot_y, c=averaged_y, cmap="winter")

plt.show()