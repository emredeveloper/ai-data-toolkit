from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree_param_dist = {"max_depth": [3, 5, 7]}

# KFOLD Grid Search
kf = KFold(n_splits = 10)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv = kf)
tree_grid_search_kf.fit(X_train, y_train)
print("KF En iyi paramter: ", tree_grid_search_kf.best_params_)
print("KF En iyi acc: ", tree_grid_search_kf.best_score_)

# LOO
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv = loo)
tree_grid_search_loo.fit(X_train, y_train)
print("LOO En iyi paramter: ", tree_grid_search_loo.best_params_)
print("LOO En iyi acc: ", tree_grid_search_loo.best_score_)