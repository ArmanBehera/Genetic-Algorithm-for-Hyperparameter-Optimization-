import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import time


dictionary = {
    "n_estimators": [150, 300],
    "learning_rate": [0.1, 0.3],
    "max_depth": [5, 30],
    "min_samples_split": [4, 10],
    "min_samples_leaf": [4, 10],
    "max_features": [0.1, 0.9],
    "loss": ["huber", "squared_error", "quantile"]
}

dict2 = {
    "n_estimators": 175,
    "learning_rate": 0.2,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 8,
    "loss": "lad"
}
obj = GenHyperOptimizer(model=ensemble.GradientBoostingRegressor(), hyperparameters=dictionary, fitnessFunction=mean_absolute_error, objective="min")

obj._uniformFillPopulation()
print(obj._odd_generation)