import services
from GenHyperOptimizer import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import time

model = linear_model.ElasticNet(
    alpha=0.718,
    l1_ratio=1.0,
    fit_intercept=True,
    max_iter=4648,
    selection='cyclic'
)

model2 = linear_model.ElasticNet(
    alpha=0.0001,
    fit_intercept=True,
    l1_ratio=0.6271862926287598,
    max_iter=2330,
    selection='cyclic'
)

model3 = linear_model.ElasticNet(
    alpha=0.01,
    l1_ratio=0.5,
    fit_intercept=True,
    max_iter=932,
    selection='random'
)
# {'alpha': 0.01, 'l1_ratio': 0.5, 'fit_intercept': True, 'max_iter': 932, 'selection': 'random'}
# 'alpha': 0.0001, 'fit_intercept': True, 'l1_ratio': 0.3731360004156354, 'max_iter': 10000, 'selection': 'random'})

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

model2.fit(X_train, y_train)

print(mean_absolute_error(y_test, model2.predict(X_test)))