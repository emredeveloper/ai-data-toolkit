from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=1000, factor = 0.5, noise = 0.08, random_state=42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])

dbscan = DBSCAN(eps = 0.15, min_samples = 15)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
plt.title("DBSCAN sonuclari")
