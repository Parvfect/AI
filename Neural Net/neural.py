
"""
So what I know about it is there are layers of thingies which each have a 
specific weight to them - right, and through backpropogation the weights 
get adjusted based on the error and stuff, hmm so maybe an array of Nodes
for each layer and each layer is connected to every single node in the other array"""

import random


class Net:
    
    layers = []

    def __init__(self, no_layers, input_size, middleware_size, output_size):
        
        layers.append(Layer(input_size, middleware_size))
        
        for i in no_layers:
            layers.append(Layer(middleware_size, middleware_size))

        layers.append(Layer(middleware_size, output_size))


class Layer:
    
    nodes = []

    def __init__(self, input_size, output_size):
        for i in range(input_size):
            nodes.append(Node())

            #Initialize weights
            nodes[i].set_weight(random.randrange(0,1))


class Node:
    #Each node has a weight and an array of node connections and the layer number
    weight = 0.0
    connection = []

    def __init__(self):
        weight = 0
    
    def set_weight(self, val):
        self.weight = val
    
    def weight(self):
        return weight 