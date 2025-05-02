# https://archive.ics.uci.edu/dataset/45/heart+disease

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

heart_disease = fetch_ucirepo(name = "heart_disease")

df = pd.DataFrame(data = heart_disease.data.features)

df["target"] = heart_disease.data.targets

# drop missing value
if df.isna().any().any():
    df.dropna(inplace = True)
    print("nan")
    
X = df.drop(["target"], axis = 1).values
y = df.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

log_reg = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=100)
log_reg.fit(X_train, y_train)

accuracy = log_reg.score(X_test, y_test)
print("Logistic Regression Acc:", accuracy)

























