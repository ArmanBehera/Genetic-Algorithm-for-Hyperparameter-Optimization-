import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn import linear_model
import time

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
    
    model = linear_model.ElasticNet(
        alpha= 0.001,
        fit_intercept= True,
        l1_ratio=0.6,
        max_iter= 3447,
        selection= 'random'
    )
    
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Time taken to train the model: {end - start}")
    
    start = time.time()
    print("1")
    print(mean_absolute_error(y_train, model.predict(X_train)))
    print(mean_absolute_error(y_test, model.predict(X_test)))
    end = time.time()
    
    print(f"Time taken to get fitness values: {end - start}")