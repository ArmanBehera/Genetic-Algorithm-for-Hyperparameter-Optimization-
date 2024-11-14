from GenHyperOptimizer import GenHyperOptimizer
from sklearn.linear_model import ElasticNet
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
from sklearn.datasets import fetch_california_housing
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint    

# Retrieving the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

del df['Latitude']
del df['Longitude']

df.dropna(axis=0, how='any', subset=None, inplace=True)

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

Grid_Search_Hyperparameter_Range = {
    "alpha": [0.001, 0.01, 0.1, 1],
    "l1_ratio": [0.001, 0.25, 0.5, 0.75, 0.999],
    "fit_intercept": [True, False],
    "max_iter": [1000, 2000, 3000, 4000, 5000],
    "selection": ['cyclic', 'random'],
    "tol": [1e-5, 1e-4, 1e-3, 1e-2],
    "precompute": [True, False],
    "warm_start": [True, False],
    "positive": [True, False],
    "copy_X": [True, False],
}

# Time for Grid Search
start = time.time()
grid_optimizer = GridSearchCV(
    estimator=ElasticNet(),
    param_grid=Grid_Search_Hyperparameter_Range,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=1,
    verbose=1
)
grid_optimizer.fit(X_train, y_train)
gridSearchTime = time.time() - start
gridSearchParams = grid_optimizer.best_params_
print(f"Time taken for Grid Search to optimize model: {gridSearchTime}")

report_data = [
    f"Time taken for Grid Search to optimize model: {gridSearchTime:.4f} seconds",
    f"Grid Search Best Parameters: {gridSearchParams}",
    ""
]

# Write the data to a file
with open("f-1-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")