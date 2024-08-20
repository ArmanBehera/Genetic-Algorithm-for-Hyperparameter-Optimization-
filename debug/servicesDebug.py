import services

floatNum = 0.0023
binaryStr = "010101010111101"
intNum = 2314

hyperparametersRange = {
    "alpha": [-0.101, 1.000],
    "l1_ratio": [-0.51, 1.00],
    "fit_intercept": [True, False],
    "max_iter": [-1500, 4000],
    "selection": ['cyclic', 'random']
}

hyperparameters = {
    "alpha": -0.001,
    "l1_ratio": -0.11,
    "fit_intercept": True,
    "max_iter": -1500,
    "selection": 'random'
}

print(f"Calculate Precision: {services._calculatePrecision(floatNum)}")

binaryLength = services._calculateBinaryLength(intNum)
print(f"Calculate Binary Length: {services._calculateBinaryLength(intNum)}")

positiveIntToBinary = services._convertIntToBinary(intNum, positive=True, length=(binaryLength))
print(f"Positive Convert int to binary: {positiveIntToBinary}")
print(f"Positive Length of binary string: {len(positiveIntToBinary)}")
print(f"Positive Convert binary to int: {services._convertBinaryToInt(positiveIntToBinary, positive=True)}")

negativeIntToBinary = services._convertIntToBinary(intNum * -1, positive=False, length=(binaryLength))
print(f"Negative Convert int to binary: {negativeIntToBinary}")
print(f"Negative Length of binary string: {len(negativeIntToBinary)}")
print(f"Negative Convert binary to int: {services._convertBinaryToInt(negativeIntToBinary, positive=False)}")

info = services.getInfo(hyperparameters=hyperparametersRange)
print(f"Get Info: {info}")

encoded_string = services.encode(parameters=hyperparameters, info=info, selection=['cyclic', 'random'])
print(f"Encoded String: {encoded_string}")

decoded_string = services.decode(chromosome=encoded_string, info=info, selection=['cyclic', 'random'])
print(f"Decoded String: {decoded_string}")
