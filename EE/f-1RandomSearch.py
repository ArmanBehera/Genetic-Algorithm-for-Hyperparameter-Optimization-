from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
    "n_estimators": randint(150, 301),
    "learning_rate": uniform(0.1, 0.2),  # range: [0.1, 0.3) -> uniform(loc=0.1, scale=0.2)
    "max_depth": randint(5, 31),
    "min_samples_split": randint(4, 11),
    "min_samples_leaf": randint(4, 11),
    "max_features": uniform(0.1, 0.8),  # range: [0.1, 0.9) -> uniform(loc=0.1, scale=0.8)
    "loss": ["huber", "squared_error", "quantile"]
}

start = time.time()
random_optimizer = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(),
    param_distributions=Random_Search_Hyperparameter_Range,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    cv=3,
    n_jobs=3,
    verbose=1,
    random_state=42
)
random_optimizer.fit(X_train, y_train)
randomSearchTime = time.time() - start
randomSearchParams = random_optimizer.best_params_
print(f"Time taken for Random Search to optimize model: {randomSearchTime}")

model = GradientBoostingRegressor(**randomSearchParams)
model.fit(X_train, y_train)

trainAccuracy = mean_absolute_error(y_train, model.predict(X_train))
testAccuracy = mean_absolute_error(y_test, model.predict(X_test))

report_data = [
    f"Time taken for Random Search to optimize model: {randomSearchTime:.4f} seconds",
    f"Random Search Best Parameters: {randomSearchParams}",
    f"Accuracy on Training dataset: {trainAccuracy}",
    f"Accuracy on Testing dataset: {testAccuracy}",
    ""
]

# Write the data to a file
with open("f-1-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")