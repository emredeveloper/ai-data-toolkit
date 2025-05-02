from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1)

X = mnist.data
y = mnist.target.astype(int)

lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c = y, cmap = "tab10", alpha = 0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="Digits")

# %% LDA vs PCA
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

colors = ["red", "blue", "green"]

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color = color, alpha = 0.8, label = target_name)
plt.legend()
plt.title("PCA of Iris Dataset")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color = color, alpha = 0.8, label = target_name)
plt.legend()
plt.title("LDA of Iris Dataset")

































