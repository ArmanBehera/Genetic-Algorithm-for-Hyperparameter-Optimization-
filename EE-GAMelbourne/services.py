import random


def _calculatePrecision(num):
    '''
        Calculates the precision (number of decimal digits) for a floating point number
    '''
    
    string = str(num).split('.')
    
    if len(string) > 1:
        # Scientific Notation with two parts. Ex: ['1', '3e-08']
        if 'e' in string[1]:
            beforeE, afterE = string[1].split('e')
            
            return (len(beforeE) + abs(int(afterE)))
    
        # For floating numbers. Ex: ['1', '1231']
        else:
            try:
                return len(string[1])
            except:
                return 0

    # Scientific Notation that has only one part. Ex: ['3e-08']
    else:
        if 'e' in string[0]:
            beforeE, afterE = string[0].split('e')
            
            return abs(int(afterE))
        else:
            return 0

def _calculateBinaryLength(value: int):
    '''
        Calculates the length of a binary value for an integer number
    '''
    return len(f"{value:b}")


def _convertIntToBinary(value: int, positive: bool, length: int):
    '''
        If the value is positive, the first bit represents the highest exponent order of the binary value
        If the value is negative, the first bit represents if the value is positive or not
    '''
    if positive:
        if value < 0:
            return ValueError("A positive value variable cannot be negative.")
        binary = f"{value:b}"
        binary = '0' * (length - len(binary)) + binary
    else:
        if value >= 0:
            binary = f"0{value:b}"
        else:
            b1 = f"{value:b}"
            binary = f"1{b1[1:]}" # Converting a negative value
    
        binary = binary[0] + '0' * (length - len(binary)) + binary[1:] 

    return binary


def _convertBinaryToInt(value: str, positive: bool):
    
    if positive:
        num = int(value, 2)
    else:
        num = int(value[1:], 2)
        if value[0] == "1":
            num *= -1
        
    return num


def getInfo(hyperparameters: dict):
    info = {}
    
    for key, value in hyperparameters.items():
        val = value[0]
        
        # For integers, the number of bits requried is calculated + 1. First bit represents if the number is positive or negative
        if type(val) == int:
            val2 = value[1]
            n = _calculateBinaryLength(val2)
            
            if val < 0 and val2 < 0:
                info[key] = ["int", n, 0, False]
            elif val < 0 and val2 >= 0:
                info[key] = ["int", n + 1, 0, False] # n + 1 to accomodate for the extra bit that represents the sign of the value
            else: # Both val and val2 are greater than zero
                info[key] = ["int", n, 0, True] 
            
        # For floating point values
        elif type(val) == float:
            val2 = value[1]
            
            p1 = _calculatePrecision(val)
            p2  = _calculatePrecision(val2)
            
            max = p1 if p1 > p2 else p2
            
            val2 = val2 * (10**max)
            n = _calculateBinaryLength(int(val2))
            
            if val < 0 and val2 < 0:
                info[key] = ["float", n, max, False]
            elif val < 0 and val2 >= 0:
                info[key] = ["float", n + 1, max, False] # n + 1 to accomodate for the extra bit that represents the sign of the value
            else: # Both val and val2 are greater than zero
                info[key] = ["float", n, max, True] 
            
        elif type(val) == str:
            n = _calculateBinaryLength((len(value) - 1))
            info[key] = ["str", n, len(value)]
        
        elif type(val) == bool:
            info[key] = ["bool", 1]
        else:
            return ValueError("Incorrect datatype passed in the values for hyperparameters.")
        
        (f"Info: {info}")
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
        chromosome = ""
        for key, value in parameters.items():
            
            data = info[key]
            datatype = data[0]
            length = data[1]
            
            if datatype == "int" or datatype == "float":
                exponent = data[2]
                positive = data[3]
                value = int(value * (10 ** exponent))
                chromosome += _convertIntToBinary(value=value, length=length, positive=positive)
                
            elif datatype == "str":
                listOfItems = kwargs[key]
                n = listOfItems.index(value)
                chromosome += _convertIntToBinary(value=n, length=length, positive=True) 
            
            elif datatype == "bool":
                chromosome += '1' if value else '0'
                
            else:
                return ValueError("Incorrect datatype passed in the values for hyperparameters.")
            
        return chromosome


def decode(chromosome="", info={}, **kwargs):
    
    hyperparameters = {}
    start = 0

    for key, value in info.items():
        
        datatype = value[0]
        length = value[1]
        end = start + length
        string = chromosome[start: end]
        start = end
        
        if datatype == "int":
            positive = value[3]
            hyperparameters[key] = _convertBinaryToInt(value=string, positive=positive)
            
        elif datatype == "float":
            exponent = value[2]
            positive = value[3]
            
            string = _convertBinaryToInt(value=string, positive=positive)
            hyperparameters[key] = (float(string) / (10 ** exponent))
            
        elif datatype == "str":
            listOfItems = kwargs[key]
            try:
                hyperparameters[key] = listOfItems[_convertBinaryToInt(value=string, positive=True)]
            except:
                hyperparameters[key] = "Invalid_data!"

        elif datatype == "bool":
            hyperparameters[key] = True if string else False
            
        else:
            return ValueError("Incorrect datatype passed in the values for hyperparameters.")
        
    return hyperparameters


def flip(p):
    '''
        Returns a boolean true value with a specified probability (a bernoulli random variable) 
        A biased coin is tossed that comes up head with probability 'p'
        Taken from book by David E. Goldberg pg 60 - 70
    '''
    return random.random() < p



def generateRandomFloat(lower_limit, higher_limit, precision):
    
    scale_factor = 10 ** precision
    
    lower_limit = int(lower_limit * scale_factor)
    higher_limit = int(higher_limit * scale_factor)
    
    randomNum = random.randrange(lower_limit, higher_limit)
    
    return (randomNum / scale_factor)
