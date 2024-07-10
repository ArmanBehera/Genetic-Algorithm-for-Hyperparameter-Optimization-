import random
import math

def _calculatePrecision(num):
    string = str(num).split('.')
    
    return len(string[1])

def _calculateBinaryLength(value: int):
    
    if value >= 0:
        n = math.ceil(math.log2(value + 1))
    else:
        n = math.ceil(math.log2(value * -1 + 1))
        
    return (n + 1)


def _convertBinary(value: int, length: int):
    if value >= 0:
        binary = f"0{value:b}"
    else:
        b1 = f"{value:b}"
        binary = f"1{b1[1:]}"
    
    binary = binary[0] + '0' * (length - len(binary)) + binary[1:]

    return binary 


def getInfo(hyperparameters):
    info = {}
    
    for key, value in hyperparameters.items():
        val = value[0]
        
        # For integers, the number of bits requried is calculated + 1. First bit represents if the number is positive or negative
        if type(val) == int:
            val2 = value[1]
            
            n = _calculateBinaryLength(val2)
            info[key] = ["int", n]
            
        # For floating point values
        elif type(val) == float:
            val2 = value[1]
            
            p1 = _calculatePrecision(val)
            p2  = _calculatePrecision(val2)
            
            max = p1 if p1 > p2 else p2
            
            v2 = val2 * (10**max)
            n = _calculateBinaryLength(int(v2))
            
            info[key] = ["float", n, max]
            
        elif type(val) == str:
            n = _calculateBinaryLength(len(value))
            info[key] = ["str", n, len(value)]
        
        elif type(val) == bool:
            info[key] = ["bool", 1]
        else:
            return ValueError("Incorrect datatype passed in the values for hyperparameters.")
        
    return info


def encode(parameters = {}, info = {}, **kwargs):
        '''
            Decodes the hyperparameter configurations into a single string
            The entire string is mapped using the index of the list that is provided by the user 
            For integers, (fixed-point coding) they are simply converted to a signed binary string, if the first bit is 0 it's positive and vice-versa
            For floats, the chosen method is scaled-integer coding
            For string, each value will be given a number and that number will be decoded to a unsigned binary string
            For booleans, one bit will be assigned. 1 is true and 0 is false.
            
            For strings, the options should be passed which will passed in kwargs
        '''
        chromsome = ""
        for key, value in parameters.items():
            
            data = info[key]
            datatype = data[0]
            length = data[1]
            
            if datatype == "int":
                chromsome += _convertBinary(value, length)
            
            elif datatype == "float":
                exponent = data[2]
                val = value * (10 ** exponent)
                chromsome += _convertBinary(int(val), length)
                
            elif datatype == "str":
                length = data[2]
                listOfItems = kwargs[key]
                print(listOfItems)
                n = listOfItems.index(value)
                chromsome += _convertBinary(n, length) 
            
            elif datatype == "bool":
                chromsome += 1 if value else 0
                
            else:
                return ValueError("Incorrect datatype passed in the values for hyperparameters.")
            
            print(value)
            print(chromsome)
            
        return chromsome

def printGenerationReport():
    # To generate a report based on the statistics calculated
    pass
    

def calculateStatistics():
    # To calculate sum_fitness, min_fitness, max_fitness
    pass


def flip(p=0.7):
    '''
        Returns a boolean true value with a specified probability (a bernoulli random variable) 
        A biased coin is tossed that comes up head with probability 'p'
        Taken from book by David E. Goldberg pg 60 - 70
    '''
    num = random.random()
    
    if num < p:
        return True
    return False