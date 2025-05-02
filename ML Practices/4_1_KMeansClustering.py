from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ornek Veri")

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "X")
plt.title("K-Means")