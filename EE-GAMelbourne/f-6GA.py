from GenHyperOptimizerImproved import GenHyperOptimizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time

# Retrieving the dataset
df = pd.read_csv('datasets/Melbourne_housing_FULL.csv')
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']

df.dropna(axis=0, how='any', subset=None, inplace=True)
df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])
X = df.drop('Price', axis=1)
y = df['Price']

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

genOptimizer = GenHyperOptimizer(model=GradientBoostingRegressor, search_space=GA_Hyperparameter_Range, scoring=mean_absolute_error, objective="min", max_pop=60, max_gen=50, iteration_number=6, elitism_rate=0.05, reduction_rate=0.0)
gaParameters = genOptimizer.optimize(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, alpha_rank=0.8, beta_rank=1.2)

gaTime = time.time() - start

print(f"Time taken for GA to optimize model: {gaTime}")

model = GradientBoostingRegressor(**gaParameters)
model.fit(X_train, y_train)
trainAccuracy = mean_absolute_error(y_train, model.predict(X_train))
testAccuracy = mean_absolute_error(y_test, model.predict(X_test))


report_data = [
    f"GA Configuration: Population Size: 60\n Number of generations: 50\n Alpha Rank: 0.8\n Beta Rank: 1.2\n Elitism Rate: 0.05\n Reduction Rate: 0.0",
    f"GA Search Space: {GA_Hyperparameter_Range}",
    f"Time taken for GA to optimize model: {gaTime:.4f} seconds",
    f"GA Best Parameters: {gaParameters}",
    f"Accuracy on Training dataset: {trainAccuracy}",
    f"Accuracy on Testing dataset: {testAccuracy}",
    ""
]

# Write the data to a file
with open("GA-melbourne-report.txt", "a") as file:
    for line in report_data:
        file.write(line + "\n")