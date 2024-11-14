from GenHyperOptimizerImproved import GenHyperOptimizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
from sklearn.datasets import fetch_california_housing

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

GA_Hyperparameter_Range = {
    "n_estimators": [100, 600],
    "learning_rate": [0.01, 0.3],
    "max_depth": [5, 50],
    "min_samples_split": [2, 20],
    "min_samples_leaf": [2, 20],
    "max_features": [0.05, 1.0],
    "loss": ["huber", "squared_error", "quantile"]
}
start = time.time()

genOptimizer = GenHyperOptimizer(model=GradientBoostingRegressor, search_space=GA_Hyperparameter_Range, scoring=mean_absolute_error, objective="min", max_pop=200, max_gen=30, iteration_number=7, elitism_rate=(6.0 / 200), reduction_rate=(2.0 / 200))
gaParameters = genOptimizer.optimize(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, alpha_rank=0.4, beta_rank=1.6)

gaTime = time.time() - start

print(f"Time taken for GA to optimize model: {gaTime}")

model = GradientBoostingRegressor(**gaParameters)
model.fit(X_train, y_train)
trainAccuracy = mean_absolute_error(y_train, model.predict(X_train))
testAccuracy = mean_absolute_error(y_test, model.predict(X_test))


report_data = [
    f"GA Configuration: Population Size: 200\n Number of generations: 30\n Alpha Rank: 0.4\n Beta Rank: 1.6\n Elitism Rate: 6/200\n Reduction Rate: 2/150",
    f"GA Search Space: {GA_Hyperparameter_Range}",
    f"Time taken for GA to optimize model: {gaTime:.4f} seconds",
    f"GA Best Parameters: {gaParameters}",
    f"Accuracy on Training dataset: {trainAccuracy}",
    f"Accuracy on Testing dataset: {testAccuracy}",
    "",
    ""
]

# Write the data to a file
with open("GA-california-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")