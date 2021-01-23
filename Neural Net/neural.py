
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

    def __init__(self, input_size, number):
        for i in range(input_size):
            nodes.append(Node(input_size, []))


class Node:

    #Each node has an array of weights and an array of inputs
    weights = []
    inputs = []

    #Make sure that inputs is an array of Nodes, would have been cooler in java
    def __init__(self, n_weights, inputs, first):
        
        self.inputs = inputs

        #initial state is off
        self.state = 0
        self.first = first

        #Randomly initializing the weights
        for i in range(n_weights):
            weights.append(random.randint(0,10000)/10000)

    def next_state(activation):
        
        z = 0

        for i in len(weights){
            
            if(self.first):
                z += inputs[i] * weights[i]
            else:
                z += inputs[i].state * weights[i]
        }

        #Update state based on inputs, weights and the activation function
        self.state = activation(z)

        return self.state

    #Create a list of random nodes
    def list_nodes():

    