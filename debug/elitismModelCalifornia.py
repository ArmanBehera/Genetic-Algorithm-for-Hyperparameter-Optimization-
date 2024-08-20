import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing
from GenHyperOptimizer import GenHyperOptimizer
import time
from skopt import BayesSearchCV

data = fetch_california_housing(as_frame=True)
df = data.frame

del df['Latitude']
del df['Longitude']

df.dropna(axis=0, how='any', subset=None, inplace=True)

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

'''
    hyperparameterRange = {
        "n_estimators": [150, 300],
        "learning_rate": [0.1, 0.3],
        "max_depth": [5, 30],
        "min_samples_split": [4, 10],
        "min_samples_leaf": [4, 10],
        "max_features": [0.1, 0.9],
        "loss": ["huber", "squared_error", "quantile"]
    }

    

    optimizer = GenHyperOptimizer(model=ensemble.GradientBoostingRegressor, search_space=hyperparameterRange, scoring=mean_absolute_error, objective="min", max_pop=50, max_gen=15, iteration_number=1, elitism_rate=0.05)

    optimizer.optimize(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, alpha_rank=0.2, beta_rank=1.8)

    end = time.time()

    print(f"Time Taken: {end - start}s")
'''

start = time.time()

hyperparameter_range = {
    "n_estimators": (150, 300),
    "learning_rate": (0.1, 0.3),
    "max_depth": (5, 30),
    "min_samples_split": (4, 10),
    "min_samples_leaf": (4, 10),
    "max_features": (0.1, 0.9),
    "loss": ("huber", "squared_error", "quantile")
}

# Initialize the Gradient Boosting Regressor model
model = ensemble.GradientBoostingRegressor()

# Configure the BayesSearchCV
optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=hyperparameter_range,
    n_iter=50,        # Number of parameter settings that are sampled
    cv=5,             # Number of cross-validation folds
    verbose=1        # Verbosity level
)

# Example: Fit the optimizer to your training data
optimizer.fit(X_train, y_train)

print("Best parameters found: ", optimizer.best_params_)

end = time.time()
print(f"Time taken: {end - start}s")