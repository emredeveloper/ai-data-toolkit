import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_pickle("4_5_GozetimsizOgrenmeDurumCalismasi")
X = data.values

X[:, 0] = np.abs(2*min(X[:, 0])) + X[:, 0]
X[:, 1] = np.abs(2*min(X[:, 1])) + X[:, 1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, alpha = 0.7, edgecolors="k")
plt.title("Musteri Segmentasyonu")
plt.xlabel("income")
plt.ylabel("spending score")

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

plt.figure(figsize = (15,6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, s = 50, alpha = 0.7, edgecolors="k")
plt.title("Kmean - Musteri Segmentasyonu")
plt.xlabel("income")
plt.ylabel("spending score")

linkage_matrix = linkage(X, method = "ward")

plt.subplot(1, 2, 2)
dendrogram(linkage_matrix)
plt.title("Dendrogram - Musteri Segmentasyonu")
plt.xlabel("Veri Noktari")
plt.ylabel("Uzaklik")























