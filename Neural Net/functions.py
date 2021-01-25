
import math

def sigmoid(x):
    return (1/(1 + math.exp(- x)))

def regression(expected, actual):
    
    sum = 0
    for _ in range(len(expected)):
        sum += math.sqrt((actual - expected) ^ 2)
    
    return sum/len(actual)