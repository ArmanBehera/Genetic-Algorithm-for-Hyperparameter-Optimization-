model=ensemble.GradientBoostingRegressor
search_space=hyperparameterRange, hyperparameterRange = {
    "n_estimators": [150, 300],
    "learning_rate": [0.1, 0.3],
    "max_depth": [5, 30],
    "min_samples_split": [4, 10],
    "min_samples_leaf": [4, 10],
    "max_features": [0.1, 0.9],
    "loss": ["huber", "squared_error", "quantile"]
}

scoring=mean_absolute_error 
objective="min"
max_pop=50
max_gen=15
iteration_number=1
elitism_rate=0.05

Best hyperparameters found: {'n_estimators': 170, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 9, 'max_features': 0.8, 'loss': 'huber'}
Accuracy: 0.45123510616167417
Time Taken: 10231.65432047844s = 2.842 hours

Trial and error best solution: 0.4684200111209944

% improvement in accuracy from trial and error: 3.66869%
% improvement in accuracy from best chromosome found in the first generation: 2.29542%

Time Taken for BayesSearchCV: 2944.2858061790466s
Best Hyperparameters found: {'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 5, 'max_features': 0.9, 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 150}
Accuracy: 0.4546677536922919
% improvement in accuracy of GA over BayesSearch: 0.75%