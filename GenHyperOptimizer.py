from sklearn.metrics import mean_absolute_error
import random

class GenHyperOptimizer:
    '''
        A genetic algorithm that optimizes the hyperparameters of a machine learning model
    '''
    
    # Arbitrary values given, to be refined
    _CROSSOVER_RATE = 0.7
    _MUTATION_RATE = 0.01
    _MAX_POP = 10
    
    _sum_fitness = 0
    _max_fitness = 0
    _min_fitness = 0
    
    _fitness = []
    _inter_fitness = []
    
    _optimized_parameters = {}
    _accuracy = 0 # Accuracy of the optimized hyperparameters
    
    _gen_count = 0
    
    _odd_generation = []
    _even_generation = []
    
    # Can implement the method elitist selection, niche and speciation
    
    # Initializes the optimizer, giving the required values
    def __init__(self, model=None, hyperparameters=None, fitnessFunction=None, cost=None):
        '''
            model: A machine learning model defined in scikit-learn
            hyperparameters: A dictionary specifying the hyperparameters to be optimized for the model. Format is given belw
                - For real values. Note: the values passed must be int or float    
                    - 'name_of_hyperparameter': [lower_bound, higher_bound]
                - For hyperparameters with string values. Note: All values that must be evaluated has to be passed
                    - 'name_of_hyperparameter': ['string_alternatives']
                - For hyperparameters with boolean values. Can pass both values unless absolutely sure. Won't impact performance much.
                    - 'name_of_hyperparameter': [True, False] 
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
    
    
    def _encode(self):
        '''
            Decodes the hyperparameter configurations into a single string
            The entire string is mapped using a fixed-point coding
            For integers, (fixed-point coding) they are simply converted to a signed binary string, if the first bit is 0 it's positive and vice-versa
            For floats, the chosen method is scaled-integer coding
            For string, each value will be given a number and that number will be decoded to a unsigned binary string
            For booleans, one bit will be assigned. 1 is true and 0 is false.
            
            All data to be normalized. to check 
        '''
        for key, value in self.hyperparameters.items():
            
            if type(value[0]) == int:
                pass
            elif type(value[0]) == float:
                pass
            elif type(value[0]) == str:
                pass
            elif type(value[0]) == bool:
                pass
            else:
                return ValueError("Incorrect datatype passed in the values for hyperparameters.")
    
    
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
        '''
            Consider two-point crossover
        '''
        pass
    
    
    def _mutation(self):
        '''
            Thinking of bit-flip mutation but to look at other methods
        '''
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