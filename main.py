import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import time

if __name__ == "__main__":
    
    start = time.time()
    
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
        "n_estimators": 75,
        "learning_rate": 0.2,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 8,
        "loss": "lad"
    }
    
    
    
    obj = GenHyperOptimizer(model=ensemble.GradientBoostingRegressor, hyperparameters=dictionary, fitnessFunction=mean_absolute_error, cost="min")
    
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
    
    obj.optimize(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    end = time.time()
    
    print(f"\nTime Taken: {end - start}")