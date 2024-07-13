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
    _CROSSOVER_RATE = 1
    _MUTATION_RATE = 0.01
    _MAX_POP = 30
    _UNIFORM_CROSSOVER_RATE = 0.5
    _MAX_GEN = 10
    
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
    
    _stringHyper = {}
    
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
    
    
    def _single_point_crossover(self, p1, p2):
        '''
            Single-point crossover
            Failed: Not creating enough diversity
        '''

        
        crossover_point = random.randrange(1, (len(p1) - 1))
        print(crossover_point)
        
        c1 = p1[:crossover_point] + p2[crossover_point:]
        c2 = p2[:crossover_point] + p1[crossover_point:]
        
        '''print(p1)
        print(p2)
        print(c1)
        print(c2) '''
        
        return c1, c2
    
    
    def _uniform_crossover(self, p1, p2):
        '''
            Uniform crossover
            Failed: Not enough diversity as most of the bits are similar
        '''
        c1 = ""
        c2 = ""
        for i in range(len(p1)):
            
            if flip(p=self._UNIFORM_CROSSOVER_RATE):
                c1 += p1[i]
                c2 += p2[i]
            else:
                c1 += p2[i]
                c2 += p1[i]

        return c1, c2
    
    
    def _hybrid_crossover(self, p1, p2):
        '''
            Here each hyperparameter value will go through a crossover
            Taken inspiration from value-encoding crossover and multi-point crossover
        '''

        start = 0
        if flip(self._CROSSOVER_RATE):
            c1 = ""
            c2 = ""
            for value in self._info.values():
            
                length = value[1]
                end = start + length
                p1_hp = p1[start: end]
                p2_hp = p2[start: end]
                start = end
                
                gc1, gc2 = self._single_point_crossover(p1_hp, p2_hp)
                
                c1 += gc1
                c2 += gc2
                
            return c1, c2
        
        else:
            return p1, p2    
                
    
    def _mutation(self, p1):
        '''
            Thinking of bit-flip mutation but to look at other methods
        '''
        # If the bit is to be changed
        length = len(p1)

        if flip(length * self._MUTATION_RATE):
            bitPosition = random.randrange(0, length)
            
            flippedBit = '0' if p1[bitPosition] == '1' else '1'
            p1 = p1[:bitPosition] + flippedBit + p1[bitPosition + 1:]
                
        return p1

    
    def _rank_selection(self, generationData, sort=False):
        '''
            Rank-based selection. Ranking is based on the cost function given
            Follows the linear ranking described by John Grefenstette in his paper
        '''   
        # Rank of the least fit is decided to be zero
        
        alpha_rank = 0.2 # Number of offsprings of the worst individual
        beta_rank = 1.2 # Number of offsprings of the best individual. Gigher the value, earlier the convergence 
        
        sum_probability = (alpha_rank + beta_rank) / 2.0 # Proved by the author
        
        if not sort:
            if self._cost == "max":
                generationData = sorted(generationData, key=lambda x: x[2], reverse=False)
            elif self._cost == "min":
                generationData = sorted(generationData, key=lambda x: x[2], reverse=True)
            else:
                return ValueError("The value for cost function can only be max and min")
            
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
        
        # If the list is not sorted, return the sorted list
        if not sort:
            return generationData, p1, p2
        
        if sort:
            return p1, p2
        
    
    def _printGenerationReport(self, generation, filename):
    # To generate a report based on the statistics calculated
        self._calculateStatistics()
        
        statistics = (
            f"Generation Count: {self._gen_count}\n"
            f"Highest Fitness: {self._max_fitness}\n"
            f"Lowest Fitness: {self._min_fitness}\n"
            f"Average Fitness: {self._sum_fitness / float(self._MAX_POP)}\n"
            f"Generation Data: {generation}"
        )
            
        with open(filename, 'w') as file:
            file.write(statistics)


    def _calculateStatistics(self, generation):
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
        
        model.fit(self._X_train, y_train)
        
        return self._fitnessFunction(y_test, model.predict(X_test))
    
    
    def _fillPopulation(self):
        '''
            Fills the first generation of population with random hyperparameters
            Next update: To make it in such a way that the points are spread out evenly 
        '''
        for i in range(self._MAX_POP):
            hyperparameters = {}
    
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
                    self._stringHyper[key] = value
                    
                    hyperparameters[key] = value[random.randrange(0, len(value))]
                
                elif datatype == "bool":
                    if (len(value)) == 1:
                        hyperparameters[key] = value[0]
                    else:
                        hyperparameters[key] = flip(p=0.5)
                
                else:
                    raise ValueError("Incorrect datatype passed in the values for hyperparameters.")  
            
            chromosome = encode(parameters=hyperparameters, info=self._info, **self._stringHyper)
            fitnessScore = self._calculateFitness(hyperparameters=hyperparameters, X_train=self._X_train, y_train=self._y_train, X_test=self._X_test, y_test=self._y_test)
            self._odd_generation.append([chromosome, hyperparameters, fitnessScore])
    
    
    # Stores the optimized parameters
    def optimize(self, X_train=None, y_train=None, X_test=None, y_test=None, iteration_number=1):
        
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
        self._gen_count += 1
        print("Filled initial population")
        for i in range(self._MAX_GEN + 1):
            
            if (self._gen_count % 2 == 0):
                currentGen = self._even_generation
                nextGen = self._odd_generation
            else:
                currentGen = self._odd_generation
                nextGen = self._even_generation
        
            p1, p2 =self._rank_selection(generationData=currentGen, sort=False)
            
            for index in range(0, (self._MAX_POP / 2)):
                c1, c2 = self._hybrid_crossover(p1=p1, p2=p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                
                c1_decoded = decode(chromosome=c1, info=self._info, **self._stringHyper)
                c2_decoded = decode(chromosome=c2, info=self._info, **self._stringHyper)
                
                c1_fitness = self._calculateFitness(hyperparameters=c1_decoded, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                c2_fitness = self._calculateFitness(hyperparameters=c2_decoded, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                nextGen.append([c1,c1_decoded, c1_fitness])
                nextGen.append([c2, c2_decoded, c2_fitness])
                
                p1, p2 =self._rank_selection(generationData=currentGen, sort=True)
                
            filename = f"Iteration_{iteration_number}/Gen_{self._gen_count}.txt"
            self._printGenerationReport(currentGen, filename)
            print(f"Number of generations completed: {self._gen_count}\nStats saved in: {filename}")
            
            currentGen = []
            self._gen_count += 1
        
    def get_params(self):
        pass