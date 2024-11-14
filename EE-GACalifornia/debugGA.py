from services import flip
import random

def _single_point_crossover(p1, p2):

        try:
            crossover_point = random.randrange(1, (len(p1) - 1))
        except ValueError:
            crossover_point = 1
        
        print(f"crossover_point: {crossover_point}")
        
        c1 = p2[:crossover_point] + p1[crossover_point:]
        c2 = p1[:crossover_point] + p2[crossover_point:]
        
        print(f"single c1: {c1}")
        print(f"single c2: {c2}")
        
        return c1, c2

def _hybrid_crossover(p1, p2, info):
        '''
            Here each hyperparameter value will go through crossover
            Taken inspiration from value-encoding crossover and multi-point crossover
        '''

        start = 0
        if flip(1.0):
            c1 = ""
            c2 = ""
            for value in info.values():
            
                length = value[1]
                end = start + length
                
                # Each block contains a separate hyperparameter value
                p1_hp = p1[start: end]
                p2_hp = p2[start: end]
                start = end
                
                print(f"p1_hp: {p1_hp}")
                print(f"p2_hp: {p2_hp}")
                
                gc1, gc2 = _single_point_crossover(p1_hp, p2_hp)
                c1 += gc1
                c2 += gc2
                
            return c1, c2
        else:
            return p1, p2  
        
def main():
    
    c1, c2 = _hybrid_crossover("010100001010011", "101101001001010", {"1": ["int", 4], "2": ["float", 4], "3": ["str", 7]})
    
    print(c1)
    print(c2)

if __name__ == "__main__":
    main()