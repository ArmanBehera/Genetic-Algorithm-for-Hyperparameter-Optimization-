import services
from debug.GenHyperOptimizerDebug import GenHyperOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import time

p1 = "00000000000000"
p2 = "11111111111111"


hyperparametersRange = {
    "alpha": [0.001, 1.000],
    "l1_ratio": [0.01, 0.99],
    "fit_intercept": [True, False],
    "max_iter": [-1500, 4000],
    "selection": ['cyclic', 'random']
}

hyperparameter1 = {
    "alpha": 0.001,
    "l1_ratio": 0.11,
    "fit_intercept": True,
    "max_iter": -1500,
    "selection": 'random'
}

hyperparameter2 = {
    "alpha": 0.999,
    "l1_ratio": 0.99,
    "fit_intercept": True,
    "max_iter": 4000,
    "selection": 'cyclic'
}

info = services.getInfo(hyperparametersRange)
h1_encoded = services.encode(info=info, parameters=hyperparameter1, selection=['cyclic', 'random'])
h2_encoded = services.encode(info=info, parameters=hyperparameter2, selection=['cyclic', 'random'])


obj = GenHyperOptimizer(model=linear_model.ElasticNet, search_space=hyperparametersRange, fitnessFunction=mean_absolute_error, objective="min", max_pop=100, max_gen=15)

'''
# Debug for crossover operator

c1, c2 = obj._single_point_crossover(p1, p2)
c1, c2 = obj._uniform_crossover(p1, p2)
c1, c2 = obj._hybrid_crossover(p1=h1_encoded, p2=h2_encoded, info=info)
print(f"Crossover: {c1, c2}")
print(f"Decoded c1: {services.decode(chromosome=c1, info=info, selection=['cyclic', 'random'])}")
print(f"Decoded c2: {services.decode(chromosome=c2, info=info, selection=['cyclic', 'random'])}")
'''

# Debug for selection operator
generationData = [['01101111100011001011011101000', {'alpha': 0.446, 'l1_ratio': 0.3, 'fit_intercept': False, 'max_iter': 2932, 'selection': 'cyclic'}, 322639.6770623298, 0.02, 0.02], ['01111111000101001101110111010', {'alpha': 0.508, 'l1_ratio': 0.5, 'fit_intercept': False, 'max_iter': 3549, 'selection': 'cyclic'}, 318075.0439532171, 0.03111111111111111, 0.05111111111111111], ['11001000010000101101001111000', {'alpha': 0.801, 'l1_ratio': 0.0, 'fit_intercept': True, 'max_iter': 3388, 'selection': 'cyclic'}, 305884.46652942116, 0.042222222222222223, 0.09333333333333332], ['01001101110000101100000000011', {'alpha': 0.311, 'l1_ratio': 0.0, 'fit_intercept': True, 'max_iter': 3073, 'selection': 'random'}, 294351.57094014925, 0.05333333333333333, 0.14666666666666667], ['00010010000000000110110010100', {'alpha': 0.072, 'l1_ratio': 0.0, 'fit_intercept': False, 'max_iter': 1738, 'selection': 'cyclic'}, 286434.1623603792, 0.06444444444444444, 0.2111111111111111], ['00100110100001101010001111001', {'alpha': 0.154, 'l1_ratio': 0.1, 'fit_intercept': True, 'max_iter': 2620, 'selection': 'random'}, 281287.8570610837, 0.07555555555555556, 0.2866666666666667], ['00101100111000001100011111001', {'alpha': 0.179, 'l1_ratio': 0.8, 'fit_intercept': False, 'max_iter': 3196, 'selection': 'random'}, 270906.5170780485, 0.08666666666666667, 0.37333333333333335], ['00010010110011110000010101011', {'alpha': 0.075, 'l1_ratio': 0.3, 'fit_intercept': True, 'max_iter': 4181, 'selection': 'random'}, 264419.7294228673, 0.09777777777777778, 0.47111111111111115], ['00011000001000001111101101010', {'alpha': 0.096, 'l1_ratio': 0.8, 'fit_intercept': False, 'max_iter': 4021, 'selection': 'cyclic'}, 260778.9672874633, 0.10888888888888888, 0.5800000000000001], ['00000110110011110010100101000', {'alpha': 0.027, 'l1_ratio': 0.3, 'fit_intercept': True, 'max_iter': 4756, 'selection': 'cyclic'}, 250722.06077155142, 0.12, 0.7000000000000001]]

for i in range(10):
    
    p1, p2 = obj._rank_selection(generationData=generationData, sort=True, alpha_rank=0.1, beta_rank=1.3)
    print(f"{p1} {p2}")
    