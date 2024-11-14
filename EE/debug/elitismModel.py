import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import time
from skopt import BayesSearchCV
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    
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
    
    start = time.time()
    
    
    hyperparameterRange = {
        "alpha": [0.001, 1.000],
        "l1_ratio": [0.00, 1.00],
        "fit_intercept": [True, False],
        "max_iter": [1500, 4000],
        "selection": ['cyclic', 'random']
    }
    
    obj = GenHyperOptimizer(model=linear_model.ElasticNet, search_space=hyperparameterRange, scoring=mean_absolute_error, objective="min", max_pop=80, max_gen=20, iteration_number=1, elitism_rate=0.05)
    obj.optimize(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, alpha_rank=0.3, beta_rank=1.7)
    
    
    
    '''
    hyperparameterRange = {
        "alpha": (0.001, 1.000),
        "l1_ratio": (0.01, 1.00),
        "fit_intercept": [True, False],
        "max_iter": (1500, 4000),
        "selection": ('cyclic', 'random')
    }
    
    model = linear_model.ElasticNet()
    
    optimizer = BayesSearchCV(
        estimator=model,
        search_spaces=hyperparameterRange,
        n_iter=30,
        cv=5
    )
    
    optimizer.fit(X_train, y_train)
    best_params = optimizer.best_params_
    print(best_params)
    '''
    
    end=time.time()
    print(f"Time taken: {end - start}")