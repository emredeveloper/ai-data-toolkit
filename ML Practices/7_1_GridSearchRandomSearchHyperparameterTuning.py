from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier()
knn_param_grid = {"n_neighbors": np.arange(2, 31)}
knn_grid_search = GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train, y_train)
print("KNN Grid Search Best Parameters: ", knn_grid_search.best_params_)
print("KNN Grid Search Best Accuracy: ", knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn, knn_param_grid, n_iter=10)
knn_random_search.fit(X_train, y_train)
print("KNN Random Search Best Parameters: ", knn_random_search.best_params_)
print("KNN Random Search Best Accuracy: ", knn_random_search.best_score_)
print()

# DT
tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

tree_random_search = RandomizedSearchCV(tree, tree_param_grid, n_iter=10)
tree_random_search.fit(X_train, y_train)
print("DT Random Search Best Parameters: ", tree_random_search.best_params_)
print("DT Random Search Best Accuracy: ", tree_random_search.best_score_)
print()

# SVM
svm = SVC()
svm_param_grid = {"C":[0.1, 1, 10, 100],
                  "gamma": [0.1, 0.01, 0.001, 0.0001]}

svm_grid_search = GridSearchCV(svm, svm_param_grid)
svm_grid_search.fit(X_train, y_train)
print("SVM Grid Search Best Parameters: ", svm_grid_search.best_params_)
print("SVM Grid Search Best Accuracy: ", svm_grid_search.best_score_)

svm_random_search = RandomizedSearchCV(svm, svm_param_grid, n_iter=10)
svm_random_search.fit(X_train, y_train)
print("SVM Random Search Best Parameters: ", svm_random_search.best_params_)
print("SVM Random Search Best Accuracy: ", svm_random_search.best_score_)


















































