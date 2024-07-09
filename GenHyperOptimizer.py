from sklearn.metrics import mean_absolute_error
import random

class GenHyperOptimizer:
    '''
        A genetic algorithm that optimizes the hyperparameters of a machine learning model
    '''
    
    # Arbitrary values given, to be refined
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.01
    MAX_POP = 10
    
    sum_fitness = 0
    max_fitness = 0
    min_fitness = 0
    
    fitness = []
    inter_fitness = []
    
    optimized_parameters = {}
    
    gen_count = 0
    
    # Can implement the method elitist selection
    
    # Initializes the optimizer, giving the required values
    def __init__(self, model=None, hyperparameters=None, fitnessFunction=None, cost=None):
        '''
            model: A machine learning model defined in scikit-learn
            hyperparameters: The hyperparameters to be optimized for the model
            scoring: A scoring method to calculate fitness function. Example: mean_absolute_error
            cost:
                - min: The lower the fitness, the better the model. For example, the lowest value for mean_absolute_error indicates an accurate model
                - max: The higher the fitness, the better the model. For example, the highest percentage accuracy of a model indicates a better model
        '''
        
        if not model:
            raise ValueError("A model must be passed.")
        
        if not hyperparameters:
            raise ValueError("A list of hyperparameters to be optimized must to be passed.")
        
        if not fitnessFunction:
            raise ValueError("A scoring metric must be passed.")
        
        if not cost:
            raise ValueError("Cost must be given as min or max to evaluate the value of fitness function.")
        
        # All values have been correctly passed.
        self.model = model
        self.hyperparameters = hyperparameters
        self.fitnessFunction = fitnessFunction
        self.cost = cost
    
    
    def _printGenerationReport(self):
        # To generate a report based on the statistics calculated
        pass
        
    
    def _calculateStatistics(self):
        # To calculate sum_fitness, min_fitness, max_fitness
        pass
    
    
    def _flip(self, p=0.7):
        '''
            Returns a boolean true value with a specified probability (a bernoulli random variable) 
            A biased coin is tossed that comes up head with probability 'p'
            Taken from book by David E. Goldberg pg 60 - 70
        '''
        num = random.random()
        
        if num < p:
            return True
        return False
    
    def _crossover(self):
        pass
    
    
    def _mutation(self):
        pass
    
    
    def _loss_selection(self):
        pass
    
    
    def _maximize_selection(self):
        pass
    
    # Stores the optimized parameters
    def optimize(self, X_train=None, y_train=None, X_test=None, y_test=None):
        
        if not X_train or y_train or X_test or y_test:
            raise ValueError("Proper training and testing datasets have to be provided.")
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    
    def get_params(self):
        pass