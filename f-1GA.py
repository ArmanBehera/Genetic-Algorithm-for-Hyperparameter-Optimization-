# Importing all required libraries
from GenHyperOptimizer import GenHyperOptimizer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
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

'''
For ElasticNet models
GA_Hyperparameter_Range = {
    "alpha": [1e-4, 1],
    "l1_ratio": [0.001, 0.999],
    "fit_intercept": [True, False],
    "max_iter": [1000, 5000],
    "selection": ['cyclic', 'random'],
    "tol": [1e-4, 1e-2],
    "precompute": [True, False],
    "warm_start": [True, False],
    "positive": [True, False],
    "copy_X": [True, False],
} '''

GA_Hyperparameter_Range = {
    "n_estimators": [150, 300],
    "learning_rate": [0.1, 0.3],
    "max_depth": [5, 30],
    "min_samples_split": [4, 10],
    "min_samples_leaf": [4, 10],
    "max_features": [0.1, 0.9],
    "loss": ["huber", "squared_error", "quantile"]
}

# Time for GA
start = time.time()
genOptimizer = GenHyperOptimizer(model=GradientBoostingRegressor, search_space=GA_Hyperparameter_Range, scoring=mean_absolute_error, objective="min", max_pop=60, max_gen=20, iteration_number=1, elitism_rate=0.05)
gaParameters = genOptimizer.optimize(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, alpha_rank=0.2, beta_rank=1.8)
gaTime = time.time() - start
print(f"Time taken for GA to optimize model: {gaTime}")

report_data = [
    f"Time taken for GA to optimize model: {gaTime:.4f} seconds",
    f"GA Best Parameters: {gaParameters}",
    ""
]

# Write the data to a file
with open("f-1-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")
