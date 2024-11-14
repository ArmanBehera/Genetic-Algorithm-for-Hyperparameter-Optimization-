from services import _calculatePrecision, _calculateBinaryLength, _convertIntToBinary, _convertBinaryToInt, getInfo

'''print(f'{_calculatePrecision(0.02101123)}')
print(f'{_calculatePrecision(0.00010123)}')
print(f'{_calculatePrecision(0.0000002312)}')
print(f'{_calculatePrecision(0.00000003)}')
print(f'{_calculatePrecision(0.00345)}')
print(f'{_calculatePrecision(0.0567)}')
print(f'{_calculatePrecision(12345.6789)}')
print(f'{_calculatePrecision(0)}')
print(f'{_calculatePrecision(1.0)}')
print(f'{_calculatePrecision(-0.0012)}')
print(f'{_calculatePrecision(0.123456789012345)}') '''

GA_Hyperparameter_Range = {
    "alpha": [1e-4, 1],
    "l1_ratio": [0.001, 0.999],
    "fit_intercept": [True, False],
    "max_iter": [1000, 5000],
    "selection": ['cyclic', 'random'],
    "tol": [1e-5, 1e-2],
    "precompute": [True, False],
    "warm_start": [True, False],
    "positive": [True, False],
    "copy_X": [True, False],
}

print(f"{getInfo(GA_Hyperparameter_Range)}")