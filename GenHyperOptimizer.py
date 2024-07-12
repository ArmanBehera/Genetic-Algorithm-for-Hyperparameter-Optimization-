from sklearn.metrics import mean_absolute_error
import random
from services import encode, decode, getInfo, flip, generateRandomFloat
import time
import pandas as pd

class GenHyperOptimizer:
    '''
        A genetic algorithm that optimizes the hyperparameters of a machine learning model
    '''
    
    # Arbitrary values given, to be refined
    _CROSSOVER_RATE = 0.7
    _MUTATION_RATE = 0.01
    _MAX_POP = 30
    
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
    
    bits = []
    
    
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
            raise ValueError("A list of hyperparameters to be optimized must be passed.")
        
        if not fitnessFunction:
            raise ValueError("A scoring metric must be passed.")
        
        if not cost:
            raise ValueError("Cost must be given as min or max to evaluate the value of fitness function.")
        
        # All values have been correctly passed.
        self._model = model
        self._hyperparameters = hyperparameters
        self._fitnessFunction = fitnessFunction
        self._cost = cost
        self._info = getInfo(hyperparameters=hyperparameters)
    
    
    def _crossover(self, p1, p2):
        '''
            Consider two-point crossover
        '''
        pass
    
    
    def _mutation(self):
        '''
            Thinking of bit-flip mutation but to look at other methods
        '''
        pass
    
    
    def _rank_selection(self, generationData, sort=False):
        '''
            Rank-based selection. Ranking is based on the cost function given
            Follows the linear ranking described by John Grefenstette in his paper
        '''
        
        # Rank of the least fit is decided to be zero
        
        if self._cost == "max":
            generationData = sorted(generationData, key=lambda x: x[2], reverse=False)
        elif self._cost == "min":
            generationData = sorted(generationData, key=lambda x: x[2], reverse=True)
        else:
            return ValueError("The value for cost function can only be max and min")
        
        alpha_rank = 0.2 # Number of offsprings of the worst individual
        beta_rank = 1.2 # Number of offsprings of the best individual. Gigher the value, earlier the convergence 
        
        sum_probability = (alpha_rank + beta_rank) / 2.0 # Proved by the author
        
        length = len(generationData)
        interSum = 0 # Intermediate sum used in the roulette wheel selection
        for i in range(length):
            probability = (alpha_rank + (float(i) / (length - 1)) * (beta_rank - alpha_rank)) / length # Formula is defined in the page 2 of the paper
            generationData[i].append(probability)
            interSum += probability
            generationData[i].append(interSum)
        
        p1 = generateRandomFloat(0.0, sum_probability, precision=5)
        p2 = generateRandomFloat(0.0, sum_probability, precision=5)
        
        p1_notfound = True
        p2_notfound = True
        
        index = 0
        
        while p1_notfound or p2_notfound:
            current = generationData[index][4]
                
            if p1_notfound:
                if p1 < current:
                    p1 = generationData[index][0]
                    p1_notfound = False
            
            if p2_notfound:
                if p2 < current:
                    p2 = generationData[index][0]
                    p2_notfound = False
            
            index += 1
        
        return p1, p2
        
    
    def printGenerationReport():
    # To generate a report based on the statistics calculated
        pass


    def calculateStatistics(self, generation):
        # To calculate sum_fitness, min_fitness, max_fitness
        interSum = 0
        
        for i in range(self._MAX_POP):
            current = generation[i][2]
            self._sum_fitness += current
            
            if current > self._max_fitness:
                self._max_fitness = current
            elif current < self._min_fitness:
                self._min_fitness = current
    
    
    def _calculateFitness(self, hyperparameters, X_train, y_train, X_test, y_test):
        # Can make it more efficient 
        model = self._model(
            **hyperparameters
        )
        
        model.fit(X_train, y_train)
        
        return self._fitnessFunction(y_test, model.predict(X_test))
    
    def _fillPopulation(self):
        '''
            Fills the first generation of population with random hyperparameters
            Next update: To make it in such a way that the points are spread out evenly 
        '''
        for i in range(self._MAX_POP):
            hyperparameters = {}
            stringHyper = {}
            for key, value in self._hyperparameters.items():
                data = self._info[key]
                datatype = data[0]
                
                if datatype == "int":
                    lowerLimit = value[0]
                    higherLimit = value[1]
                    
                    hyperparameters[key] = random.randrange(lowerLimit, higherLimit + 1) 
                    
                elif datatype == "float":
                    lowerLimit = value[0]
                    higherLimit = value[1]
                    
                    hyperparameters[key] = generateRandomFloat(lowerLimit, higherLimit, data[2])
                    
                elif datatype == "str":
                    stringHyper[key] = value
                    
                    hyperparameters[key] = value[random.randrange(0, len(value))]
                
                elif datatype == "bool":
                    if (len(value)) == 1:
                        hyperparameters[key] = value[0]
                    else:
                        hyperparameters[key] = flip(p=0.5)
                
                else:
                    raise ValueError("Incorrect datatype passed in the values for hyperparameters.")  
            
            chromosome = encode(parameters=hyperparameters, info=self._info, **stringHyper)
            fitnessScore = self._calculateFitness(hyperparameters=hyperparameters, X_train=self._X_train, y_train=self._y_train, X_test=self._X_test, y_test=self._y_test)
            self._odd_generation.append([chromosome, hyperparameters, fitnessScore])
    
    
    # Stores the optimized parameters
    def optimize(self, X_train=None, y_train=None, X_test=None, y_test=None):
        
        try: 
            if not X_train or y_train or X_test or y_test:
                raise ValueError("Proper training and testing datasets have to be provided.")
        except ValueError:
            if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                raise ValueError("Datasets provided cannot be empty.")
        
        print("Values are good to go!")
        
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        
        
        print("Starting to fill the initial generation of population.")
        self._fillPopulation()
        
        print(self._odd_generation)
        print()
        p1, p2 = self._rank_selection(self._odd_generation, sort=False)
         
        for i in range(15):
            print(f"p1: {p1} and p2: {p2}")
            p1, p2 = self._rank_selection(self._odd_generation, sort=True)
            
        
    
    def get_params(self):
        pass