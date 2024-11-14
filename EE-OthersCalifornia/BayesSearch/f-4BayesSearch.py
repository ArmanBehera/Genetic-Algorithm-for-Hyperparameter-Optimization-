from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
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


from skopt.space import Real, Integer, Categorical

Bayes_Search_Hyperparameter_Range = {
    "n_estimators": Integer(100, 600),
    "learning_rate": Real(0.01, 0.3),
    "max_depth": Integer(5, 50),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(2, 20),
    "max_features": Real(0.05, 1.0),
    "loss": Categorical(["huber", "squared_error", "quantile"])
}

start = time.time()

bayes_optimizer = BayesSearchCV(
    estimator=GradientBoostingRegressor(),
    search_spaces=Bayes_Search_Hyperparameter_Range,
    scoring='neg_mean_absolute_error',
    n_iter=150,
    cv=3,
    n_jobs=1,
    verbose=3
)

bayes_optimizer.fit(X_train, y_train)
bayesTime = time.time() - start
bayesParameters = bayes_optimizer.best_params_

print(f"Time taken for Bayes Search to optimize model: {bayesTime}")

model = GradientBoostingRegressor(**bayesParameters)
model.fit(X_train, y_train)

trainAccuracy = mean_absolute_error(y_train, model.predict(X_train))
testAccuracy = mean_absolute_error(y_test, model.predict(X_test))

report_data = [
    f"Iteration_4",
    f"Time taken for Bayes Search to optimize model: {bayesTime:.4f} seconds",
    f"Bayes Search Best Parameters: {bayesParameters}",
    f"Accuracy on Training dataset: {trainAccuracy}",
    f"Accuracy on Testing dataset: {testAccuracy}",
    f"BayesSearch Settings: n_iter: 150, cv=3, n_jobs=1"
]

print(report_data)

# Write the data to a file
with open("BayesSearchReport.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")


with open("f-4BayesSearchInfo.txt", "w") as file:
    for key, value in bayes_optimizer.cv_results_.items():
        file.write(f"{key}: {value}\n")