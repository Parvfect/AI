


class Activation:

    def sigmoid(x):
        return (1/(1 + e ^ (- x)))
    
class Loss:
    
    def regression(expected, actual):
        
        sum = 0
        for i in range(len(expected)):
            sum += sqrt((actual - expected) ^ 2)
        
        return sum/len(actual)