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

Random_Search_Hyperparameter_Range = {
    "alpha": uniform(1e-4, 1 - 1e-4),
    "l1_ratio": uniform(0.001, 0.999 - 0.001),
    "fit_intercept": [True, False],
    "max_iter": randint(1000, 5001),
    "selection": ['cyclic', 'random'],
    "tol": uniform(1e-5, 1e-2 - 1e-5),
    "precompute": [True, False],
    "warm_start": [True, False],
    "positive": [True, False],
    "copy_X": [True, False],
}

start = time.time()
random_optimizer = RandomizedSearchCV(
    estimator=ElasticNet(),
    param_distributions=Random_Search_Hyperparameter_Range,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    cv=5,
    n_jobs=1,
    verbose=1,
    random_state=42
)
random_optimizer.fit(X_train, y_train)
randomSearchTime = time.time() - start
randomSearchParams = random_optimizer.best_params_
print(f"Time taken for Random Search to optimize model: {randomSearchTime}")

report_data = [
    f"Time taken for Random Search to optimize model: {randomSearchTime:.4f} seconds",
    f"Random Search Best Parameters: {randomSearchParams}",
    ""
]

# Write the data to a file
with open("f-1-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")