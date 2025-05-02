from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

poly_feat = PolynomialFeatures(degree=2)
X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)
print("Polynomial Regression rmse: ", mean_squared_error(y_test, y_pred, squared = False))

lin_reg = LinearRegression()
lin_reg .fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Multi Variable Linear Regression rmse: ", mean_squared_error(y_test, y_pred, squared = False))
