from sklearn.metrics import mean_absolute_error
import random
from services import encode, decode, getInfo, flip, generateRandomFloat
import os
import math

class GenHyperOptimizer:
    '''
        A genetic algorithm that optimizes the hyperparameters of a machine learning model
    '''
    
    # Arbitrary values given, to be refined
    _CROSSOVER_RATE = 0.7
    _MUTATION_RATE = 0.03
    _ELITISM_RATE = 0.04
    _REDUCTION_RATE = 0.00
    
    _MAX_POP = 0
    _ELITISM_POP = 0
    _REDUCTION_POP = 0 # The number of individuals that are going to be reduced each generation
    _MAX_GEN = 0
    
    _sum_fitness = 0
    _max_fitness = 0
    _min_fitness = 0
    
    _fitness = []
    
    _best_params = {}
    _best_accuracy = 0
    
    _gen_count = 0
    
    _odd_generation = []
    _even_generation = []
    
    _stringHyper = {}
    
    # Initializes the optimizer, giving the required values
    def __init__(self, model=None, search_space=None, scoring=None, objective=None, max_pop=30, max_gen=10, elitism_rate=0.04, reduction_rate=0.00, iteration_number=None):
        '''
            model: A machine learning model defined in scikit-learn
            search_space: A dictionary specifying the hyperparameters to be optimized for the model. Format is given belw
                - For real values. Note: the values passed must be int or float    
                    - 'name_of_hyperparameter': [lower_bound, higher_bound]
                - For hyperparameters with string values. Note: All values that must be evaluated has to be passed
                    - 'name_of_hyperparameter': ['string_alternatives']
                - For hyperparameters with boolean values. Can pass both values unless absolutely sure. Won't impact performance much.
                    - 'name_of_hyperparameter': [True, False] 
            scoring: A scoring method to calculate fitness function. Example: mean_absolute_error
            objective:
                - min: The lower the fitness, the better the model. For example, the lowest value for mean_absolute_error indicates an accurate model
                - max: The higher the fitness, the better the model. For example, the highest percentage accuracy of a model indicates a better model
        '''
        
        if not model:
            raise ValueError("A model must be passed.")
        
        if not search_space:
            raise ValueError("A list of hyperparameters to be optimized must be passed.")
        
        if not scoring:
            raise ValueError("A scoring metric must be passed.")
        
        if not objective:
            raise ValueError("One of the objective functions, min or max, should be passed.")
        else:
            if objective != "min" and objective != "max":
                raise ValueError("Objective must be given as min or max to evaluate the value of fitness function.")
        
        if max_pop % 2 == 1:
            raise ValueError("Maximum population can only be even.")
        
        # All values have been correctly passed.
        self._model = model
        self._search_space = search_space
        self._scoring = scoring
        self._objective = objective
        self._info = getInfo(hyperparameters=search_space)
        self._MAX_POP = max_pop
        self._MAX_GEN = max_gen
        self._ELITISM_RATE = elitism_rate
        self._REDUCTION_RATE = reduction_rate
        self._iteration_number = iteration_number

        # Number of individuals that are to be directly shifted to the next gen without the influence of operators
        ELITISM_POP = math.ceil(elitism_rate * max_pop)
        
        # If the number of individuals is odd, it reduces 1 to make it even
        if (ELITISM_POP % 2 == 1):
            ELITISM_POP -= 1
            
        self._ELITISM_POP = ELITISM_POP
        
        REDUCTION_POP = math.ceil(reduction_rate * max_pop)
        
        if (REDUCTION_POP % 2 == 1):
            REDUCTION_POP -= 1
            
        self._REDUCTION_POP= REDUCTION_POP
    

    def _single_point_crossover(self, p1, p2):

        try:
            crossover_point = random.randrange(1, (len(p1) - 1))
        except ValueError:
            crossover_point = 1
        
        c1 = p1[:crossover_point] + p2[crossover_point:]
        c2 = p2[:crossover_point] + p1[crossover_point:]
        
        return c1, c2
    
    
    def _hybrid_crossover(self, p1, p2):
        '''
            Here each hyperparameter value will go through crossover
            Taken inspiration from value-encoding crossover and multi-point crossover
        '''

        if flip(self._CROSSOVER_RATE):
            start = 0
            c1 = ""
            c2 = ""
            for value in self._info.values():
            
                length = value[1]
                end = start + length
                
                # Each block contains a separate hyperparameter value
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
        
        for bitPosition in range(len(p1)):
            if (flip(self._MUTATION_RATE)):
                flippedBit = '0' if p1[bitPosition] == '1' else '1'
                p1 = p1[:bitPosition] + flippedBit + p1[bitPosition + 1:]

        
        return p1
    
    
    def _rank_selection(self, generationData, alpha_rank, beta_rank, intermediate=False):
        '''
            Rank-based selection. Ranking is based on the cost function given
            Follows the linear ranking described by John Grefenstette in his paper
            alpha_rank: Number of offsprings of the worst individual
            beta_rank: Number of offsprings of the best individual. 
            
            In this selection, the rank of the least fit individual is defined to be zero
        '''
        sum_probability = (alpha_rank + beta_rank) / 2.0 # Proved by the author
        
        # Checks if the intermediate sums of the probabilities have been added or not
        if not intermediate:
            length = len(generationData)
            
            interSum = 0 # Intermediate sum used in the roulette wheel selection

            for i in range(length):
                probability = (alpha_rank + ((float(i) / (length - 1)) * (beta_rank - alpha_rank))) / length # Formula is defined in the page 2 of the paper
                generationData[i].append(probability) # Do I need to include the probability of each individual being selected?
                interSum += probability
                generationData[i].append(interSum)
        
        p1 = generateRandomFloat(0.0, sum_probability, precision=5)
        p2 = generateRandomFloat(0.0, sum_probability, precision=5)
        
        p1_notfound = True
        p2_notfound = True
        
        index = 0
        
        while (p1_notfound or p2_notfound) and index < self._MAX_POP:
            current = generationData[index][4]
                
            if p1_notfound:
                if p1 < current:
                    p1 = generationData[index]
                    p1_notfound = False
            
            if p2_notfound:
                if p2 < current:
                    p2 = generationData[index]
                    p2_notfound = False
            
            index += 1
        
        return p1, p2
        
    
    def _printGenerationReport(self, generation, filename):
        # To generate a report based on the statistics calculated
        self._calculateStatistics(generation=generation)
        
        if not filename.endswith('.txt'):
            filename += '.txt'
    
        # Ensure the directory exists
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        
        statistics = (
            f"Model: {self._model}\n"
            f"Fitness Function: {self._scoring}\n"
            f"Objective: {self._objective}\n"
            f"Generation Count: {self._gen_count}\n"
            f"Highest Fitness: {self._max_fitness}\n"
            f"Lowest Fitness: {self._min_fitness}\n"
            f"Average Fitness: {self._sum_fitness / float(self._MAX_POP)}\n"
            f"Best Chromosome: {generation[(self._MAX_POP - 1)][0]}\n"
            f"Best Hyperparameter Configuration: {generation[self._MAX_POP - 1][1]}\n"
            f"Generation Data: {generation}"
        )
            
        with open(filename, 'w') as file:
            file.write(statistics)


    def _calculateStatistics(self, generation):
        # To calculate sum_fitness, min_fitness, max_fitness
        self._sum_fitness = 0
        length = len(generation)
        
        if self._objective == "max":
            self._max_fitness = generation[length - 1][2]
            self._min_fitness = generation[0][2]

            self._best_accuracy = self._max_fitness
            self._best_params = generation[length][1]
        else:
            self._max_fitness = generation[0][2]
            self._min_fitness = generation[length - 1][2]
            
            self._best_accuracy = self._min_fitness
            self._best_params = generation[length - 1][1]
        
        for i in range(self._MAX_POP):
            current = generation[i][2]
                
            self._sum_fitness += current

    
    def _calculateFitness(self, hyperparameters, X_train, y_train, X_test, y_test):
        # Can make it more efficient 
        model = self._model(
            **hyperparameters
        )
        
        model.fit(X_train, y_train)
        
        return self._scoring(y_test, model.predict(X_test))
    
    
    def _randomFillPopulation(self):
        '''
            Fills the first generation of population with random hyperparameters
            Next update: To make it in such a way that the points are spread out evenly 
        '''
        for i in range(self._MAX_POP):
            hyperparameters = {}
    
            for key, value in self._search_space.items():
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
            
        self._best_accuracy = self._odd_generation[0][2]
    
    
    def _decodeHyperparameters(self, c, p1, p2):
        '''
            Decodes chromosome into hypeparameters
            Checks for validity by making sure that the values lie within the range provided by the user.
        '''
        
        c_decoded = decode(chromosome=c, info=self._info, **self._stringHyper)
        
        for key in self._stringHyper.keys():
            if c_decoded[key] == "Invalid_data!":  # This is set in the function, decode() of services
                if flip(0.5):
                    c_decoded[key] = p1[1][key]
                else:
                    c_decoded[key] = p2[1][key]
        
        for key, value in self._info.items():
            datatype = value[0]
            if datatype == "str":
                if c_decoded[key] not in self._search_space[key]:
                    if flip(0.5):
                        c_decoded[key] = p1[1][key]
                    else:
                        c_decoded[key] = p2[1][key]
            elif datatype == "bool":
                if c_decoded[key] not in value:
                    c_decoded[key] = value[0]
            else:
                if c_decoded[key] < self._search_space[key][0] or c_decoded[key] > self._search_space[key][1]:
                    if flip(0.5):
                        c_decoded[key] = p1[1][key]
                    else:
                        c_decoded[key] = p2[1][key]
        
        c = encode(c_decoded, info=self._info, **self._stringHyper)
        return c, c_decoded
    
    
    def optimize(self, X_train=None, y_train=None, X_test=None, y_test=None, alpha_rank=0.3, beta_rank=1.7):
        '''
            Runs the GA in the appropriate order 
        '''
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
        self._randomFillPopulation()
        self._gen_count += 1
        print("Filled initial population")

        try:
            for i in range(self._MAX_GEN):
                
                # Sorting the generation data
                if (self._gen_count % 2 == 0):
                    currentGen = self._even_generation.copy()
                    self._even_generation.clear()
                    nextGen = self._odd_generation
                else:
                    currentGen = self._odd_generation.copy()
                    self._odd_generation.clear()
                    nextGen = self._even_generation

                if self._objective == "max":
                    currentGen = sorted(currentGen, key=lambda x: x[2], reverse=False)
                else:
                    currentGen = sorted(currentGen, key=lambda x: x[2], reverse=True)
                
                # Transferring the best individuals into the next generation
                # In this elitism method, even the elitist individuals are open to mating
                for i in range(1, self._ELITISM_POP + 1):
                    nextGen.append((currentGen[-i][0:3])) # Keeps the chromosome, the hyperparameter configuration and the fitness score. Removes the ranking and intermediate sum
                
                if not self._REDUCTION_POP == 0:
                    self._MAX_POP -= self._REDUCTION_POP
                    currentGen = currentGen[self._REDUCTION_POP:] # removes the worst individual from the population

                p1, p2 = self._rank_selection(generationData=currentGen, intermediate=False, alpha_rank=alpha_rank, beta_rank=beta_rank) 
                
                for index in range(int((self._MAX_POP - self._ELITISM_POP) / 2)):
                    
                    c1, c2 = self._hybrid_crossover(p1=p1[0], p2=p2[0])
                    c1 = self._mutation(c1)
                    c2 = self._mutation(c2)
                    
                    c1, c1_decoded = self._decodeHyperparameters(c=c1, p1=p1, p2=p2)
                    c2, c2_decoded = self._decodeHyperparameters(c=c2, p1=p1, p2=p2)
                    
                    
                    c1_fitness = self._calculateFitness(hyperparameters=c1_decoded, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                    c2_fitness = self._calculateFitness(hyperparameters=c2_decoded, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                    
                    nextGen.append([c1,c1_decoded, c1_fitness])
                    nextGen.append([c2, c2_decoded, c2_fitness])
                    
                    p1, p2 =self._rank_selection(generationData=currentGen, intermediate=True, alpha_rank=alpha_rank, beta_rank=beta_rank)
                
                filename = f"Iteration_{self._iteration_number}/Gen_{self._gen_count}.txt"
                self._printGenerationReport(currentGen, filename)
                print(f"Number of generations completed: {self._gen_count}\nStats saved in: {filename}")
                
                # Updating the number of elitism population every generation to account for 
                ELITISM_POP = math.ceil(self._ELITISM_RATE * self._MAX_POP)
            
                # If the number of individuals is odd, it reduces 1 to make it even
                if (ELITISM_POP % 2 == 1):
                    ELITISM_POP -= 1
                
                
                self._ELITISM_POP = 2 if ELITISM_POP == 0 else ELITISM_POP  
                
                self._gen_count += 1
            
        except Exception as e:
            print (f"Exception: {e}")
        
        finally:
            print(f"Best hyperparameters found: {self._best_params}")
            print(f"Accuracy: {self._best_accuracy}")
        
        return self._best_params
