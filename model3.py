import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import time
from skopt import BayesSearchCV

if __name__ == "__main__":
    
    start = time.time()
    
    dictionary = {
        "alpha": (0.0001, 1.0),  # Increase upper bound for alpha
        "l1_ratio": (0.0, 1.0),
        "fit_intercept": [True, False],
        "max_iter": (500, 5000),  # Increase range for max_iter
        "selection": ('cyclic', 'random')
    }
    
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
    
    model = linear_model.ElasticNet()
    
    optimizer = BayesSearchCV(
        estimator=model,
        search_spaces=dictionary,
        n_iter=30,
        cv=5
    )
    
    optimizer.fit(X_train, y_train)
    best_params = optimizer.best_params_
    print(best_params)
    
    end = time.time()
    
    print(f"\nTime Taken: {end - start}")