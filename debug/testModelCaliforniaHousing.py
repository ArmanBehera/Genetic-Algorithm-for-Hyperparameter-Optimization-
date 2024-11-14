from sklearn import ensemble
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = fetch_california_housing(as_frame=True)
df = data.frame

del df['Latitude']
del df['Longitude']

df.dropna(axis=0, how='any', subset=None, inplace=True)

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

model = ensemble.GradientBoostingRegressor(
    learning_rate= 0.1,
    loss= 'huber',
    max_depth= 5,
    max_features= 0.9,
    min_samples_leaf= 10,
    min_samples_split= 10,
    n_estimators= 150
)

model.fit(X_train, y_train)

print(mean_absolute_error(y_test, model.predict(X_test)))