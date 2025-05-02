from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# veri seti inceleme
iris = load_iris()

X = iris.data # features
y = iris.target # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# DT modeli olustur ve train et
tree_clf = DecisionTreeClassifier(criterion = "gini", max_depth = 5, random_state = 42) # criterion="entropy"
tree_clf.fit(X_train, y_train)

# DT evaluation test
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("iris veri seti ile egitilen DT modeli dogrulugu: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("conf_matrix: ")
print(conf_matrix)

plt.figure(figsize = (15,10))
plot_tree(tree_clf, filled = True, feature_names = iris.feature_names, class_names = list(iris.target_names))
plt.show()

feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse = True)
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# veri seti inceleme
iris = load_iris()

n_classes = len(iris.target_names)
plot_colors = "ryb"
    
for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    
    X = iris.data[:, pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X, y)
    
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad = 2.5)
    DecisionBoundaryDisplay.from_estimator(clf, 
                                           X, 
                                           cmap = plt.cm.RdYlBu,
                                           response_method="predict",
                                           ax=ax,
                                           xlabel=iris.feature_names[pair[0]],
                                           ylabel=iris.feature_names[pair[1]])
    
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label = iris.target_names[i], 
                    cmap = plt.cm.RdYlBu,
                    edgecolors="black")
    
plt.legend()

# %% 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes = load_diabetes()

X = diabetes.data # features
y = diabetes.target # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# karar agaci regresyon modeli
tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("mse: ",mse)

rmse = np.sqrt(mse)
print("rmse: ",rmse)

# %%
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# create a data set
X = np.sort(5 * np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 5 * (0.5 - np.random.rand(16))

# plt.scatter(X, y)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=15)
regr_1.fit(X, y)
regr_2.fit(X, y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.plot(X, y, c = "red", label = "data")
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "Max Depth: 2", linewidth = 2)
plt.plot(X_test, y_pred_2, color = "green", label = "Max Depth: 15", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()



































