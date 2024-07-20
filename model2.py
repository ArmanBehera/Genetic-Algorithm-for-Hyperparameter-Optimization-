import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import time

if __name__ == "__main__":
    
    start = time.time()
    
    dictionary = {
        "alpha": [0.001, 1.000],
        "l1_ratio": [0.00, 1.00],
        "fit_intercept": [True, False],
        "max_iter": [1500, 5000],
        "selection": ['cyclic', 'random']
    }
    
    obj = GenHyperOptimizer(model=linear_model.ElasticNet, search_space=dictionary, fitnessFunction=mean_absolute_error, objective="min", max_pop=10, max_gen=8)
    
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
    
    obj.optimize(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, iteration_number=10)
    
    end = time.time()
    
    print(f"\nTime Taken: {end - start}")