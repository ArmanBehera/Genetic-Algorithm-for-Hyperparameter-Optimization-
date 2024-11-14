from GenHyperOptimizer import GenHyperOptimizer
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
    "n_estimators": Integer(150, 300),
    "learning_rate": Real(0.1, 0.3),
    "max_depth": Integer(5, 30),
    "min_samples_split": Integer(4, 10),
    "min_samples_leaf": Integer(4, 10),
    "max_features": Real(0.1, 0.9),
    "loss": Categorical(["huber", "squared_error", "quantile"])
}

start = time.time()

bayes_optimizer = BayesSearchCV(
    estimator=GradientBoostingRegressor(),
    search_spaces=Bayes_Search_Hyperparameter_Range,
    scoring='neg_mean_absolute_error',
    n_iter=20,
    cv=5,
    n_jobs=1,
    verbose=3,
    random_state=42
)

bayes_optimizer.fit(X_train, y_train)
bayesTime = time.time() - start
bayesParameters = bayes_optimizer.best_params_

print(f"Time taken for Bayes Search to optimize model: {bayesTime}")

model = GradientBoostingRegressor(**bayesParameters)
model.fit(X_train, y_train)

testAccuracy = mean_absolute_error(y_train, model.predict(X_train))
trainAccuracy = mean_absolute_error(y_test, model.predict(X_test))

report_data = [
    f"Time taken for Bayes Search to optimize model: {bayesTime:.4f} seconds",
    f"Bayes Search Best Parameters: {bayesParameters}",
    ""
]

# Write the data to a file
with open("f-1-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")