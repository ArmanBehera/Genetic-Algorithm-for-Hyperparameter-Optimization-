import services

v1 = 257
v2 = -257

'''v1_b = services._convertIntToBinary(v1, length = 13, positive=True)
v2_b = services._convertIntToBinary(v1, length=13, positive=False)
v3_b = services._convertIntToBinary(v2, length = 13, positive=False)
v1_i = services._convertBinaryToInt(v1_b, positive=True)
v2_i = services._convertBinaryToInt(v2_b, positive=False)
v3_i = services._convertBinaryToInt(v3_b, positive=False)
print(f"{v1:b}")
print(f"{v2:b}")
print(v1_b)
print(v2_b)
print(v3_b)
print(v1_i)
print(v2_i)
print(v3_i) '''

dictionary = {
    "n_estimators": [150, 300],
    "learning_rate": [0.1, 0.3],
    "max_depth": [5, 30],
    "min_samples_split": [4, 10],
    "min_samples_leaf": [4, 10],
    "max_features": [0.1, 0.9],
    "loss": ["huber", "squared_error", "quantile"]
}

configuration = {
    "n_estimators": 200,
    "learning_rate": 0.2,
    "max_depth": 15,
    "min_samples_split": 6,
    "min_samples_leaf": 6,
    "max_features": 0.6,
    "loss": "quantile"
}

info = services.getInfo(dictionary)
encoded = services.encode(parameters=configuration, info=info, loss=dictionary['loss'])
print(info)
print(encoded)
decoded = services.decode(chromosome=encoded, info=info, loss=dictionary['loss'])
print(decoded)