from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# veri olustur
X = np.random.rand(100,1)
y = 3 + 4 * X + np.random.rand(100,1) # y = 3 + 4x 

# plt.scatter(X,y)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure()
plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), color = "red", alpha = 0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineer Regresyon")

# y = 3 + 4x  -> y = a0 + a1x
a1 = lin_reg.coef_[0][0] 
print("a1: ",a1)

a0 = lin_reg.intercept_[0]
print("a0: ",a0)

for i in range(100):
    y_ = a0 + a1 * X
    plt.plot(X, y_, color = "green", alpha = 0.7)

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np

diabetes = load_diabetes()

diabetes_X, diabetes_y = load_diabetes(return_X_y = True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20] 
diabetes_X_test = diabetes_X[-20:] 

diabetes_y_train = diabetes_y[:-20] 
diabetes_y_test = diabetes_y[-20:] 

lin_reg = LinearRegression()

lin_reg.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = lin_reg.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("mse: ", mse)
r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print("r2: ", r2)

plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")
plt.plot(diabetes_X_test, diabetes_y_pred, color = "blue")













































