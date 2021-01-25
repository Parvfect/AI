
"""
So what I know about it is there are layers of thingies which each have a 
specific weight to them - right, and through backpropogation the weights 
get adjusted based on the error and stuff, hmm so maybe an array of Nodes
for each layer and each layer is connected to every single node in the other array"""

import random
from functions import *

class Net:
    
    """
    A neural net initialzation will need the following
    1) First layer size
    2) Hidden layer size
    3) Output size
    4) Number of hidden layers
    5) Inputs are the information right my bad (every state requires an input ouch)
    """
    layers = []
    input = []

    def __init__(self, no_hidden_layers, input_size, hidden_layer_size, output_size, input):
        
        self.layers.append(Layer(input_size, True))
        self.input = input

        for i in range(no_hidden_layers):
            self.layers.append(Layer(hidden_layer_size, False))

        self.layers.append(Layer(output_size, False))

    def forward_pass(self):
        for i in self.input:
            self.layers[0].forward_pass(sigmoid, i)
            for j in range(1, len(self.layers)):
                self.layers[j].forward_pass(sigmoid, self.layers[j-1].states())


class Layer:
    
   
    def __init__(self, input_size, first):
        
        self.input_size = input_size
        self.first = first
        self.nodes = []

        for i in range(input_size):
            self.nodes.append(Node(input_size, first))
            
    # Updates the layer for one pass
    def forward_pass(self, activation, input):

        for i in self.nodes:
            i.next_state(activation, input)
    
    def states(self):
        a = []
        for i in self.nodes:
            a.append(i.state)

        return a

class Node:

    
    #Make sure that inputs is an array of Nodes, would have been cooler in java
    def __init__(self, n_weights, first):
        
        self.weights = []
        #initial state is off
        self.state = 0
        self.first = first
        self.n_weights = n_weights
        #Randomly initializing the weights
        for i in range(self.n_weights):
            self.weights.append(random.randint(0,10000)/10000)

    def next_state(self, activation, input = []):
        
        z = 0
        print(len(self.weights))
        print(len(input))
        for i in range(len(self.weights)):
            
            z += input[i] * self.weights[i]
        

        #Update state based on inputs, weights and the activation function
        self.state = activation(z)

        return self.state

    