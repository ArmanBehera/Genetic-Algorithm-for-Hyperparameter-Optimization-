from services import getInfo, _calculatePrecision, encode

if __name__ == "__main__":
    dictionary = {
        "n_estimators": [50, 300],
        "learning_rate": [0.1, 0.5],
        "max_depth": [5, 30],
        "min_samples_split": [4, 10],
        "min_samples_leaf": [4, 10],
        "loss": ["huber", "ls", "lad"]
    }
    
    dict2 = {
        "n_estimators": 75,
        "learning_rate": 0.2,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 8,
        "loss": "ls"
    }
    info = getInfo(dictionary)
    print(info)
    encode(parameters=dict2, info=info, loss=dictionary["loss"])