from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
from sklearn.datasets import fetch_california_housing
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
    "n_estimators": randint(150, 601),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(5, 51),
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(2, 21),
    "max_features": uniform(0.05, 0.9),
    "loss": ["huber", "squared_error", "quantile"]
}

start = time.time()

random_search_optimizer = RandomizedSearchCV(
    estimator = GradientBoostingRegressor(),
    param_distributions = Random_Search_Hyperparameter_Range,
    scoring = 'neg_mean_absolute_error',
    n_iter = 500,
    cv = 3,
    n_jobs = 1,
    verbose = 3
)

random_search_optimizer.fit(X_train, y_train)
randomTime = time.time() - start
randomParameters = random_search_optimizer.best_params_


print(f"Time taken for Random Search to optimize model: {randomTime}")

model = GradientBoostingRegressor(**randomParameters)
model.fit(X_train, y_train)

trainAccuracy = mean_absolute_error(y_train, model.predict(X_train))
testAccuracy = mean_absolute_error(y_test, model.predict(X_test))

report_data = [
    f"Iteration_4",
    f"Time taken for Random Search to optimize model: {randomTime:.4f} seconds",
    f"Random Search Best Parameters: {randomParameters}",
    f"Accuracy on Training dataset: {trainAccuracy}",
    f"Accuracy on Testing dataset: {testAccuracy}",
    f"Random Search setting: n_iter: 500, cv=3, n_jobs=1",
]

print(report_data)

# Write the data to a file
with open("RandomSearchReport.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")


with open("f-4RandomSearchInfo.txt", "w") as file:
    for key, value in random_search_optimizer.cv_results_.items():
        file.write(f"{key}: {value}\n")