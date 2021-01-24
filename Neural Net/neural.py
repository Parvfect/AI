
"""
So what I know about it is there are layers of thingies which each have a 
specific weight to them - right, and through backpropogation the weights 
get adjusted based on the error and stuff, hmm so maybe an array of Nodes
for each layer and each layer is connected to every single node in the other array"""

import random


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

    def __init__(self, no_hidden_layers, input_size, hidden_layer_size, output_size, input):
        
        layers.append(Layer(input_size, True))
        
        for i in range(no_hidden_layers):
            layers.append(Layer(hidden_layer_size, False))

        layers.append(Layer(output_size, False))
    
    def forward_pass():
        for i,j in input,layers:
            j.forward_pass(sigmoid, i)
    


class Layer:
    
    nodes = []

    def __init__(self, input_size, first):
        
        self.input_size = input_size
        self.number = number
        
        #The problem here is that if it is the first layer, where in the actual inputs will be user defined rather 
        # than updating from the previous layer - turns out that is the informatino that is to be mapped lmao
        if(first):
            self.initial_input = initial_input

        for i in range(input_size):
            nodes.append(Node(input_size, first)

    # Updates the layer for one pass
    def forward_pass(activation, input):

        for i in nodes:
            i.next_state(activation, input)


class Node:

    #Each node has an array of weights and an array of inputs
    weights = []
    inputs = []

    #Make sure that inputs is an array of Nodes, would have been cooler in java
    def __init__(self, n_weights, first):
        
        self.inputs = inputs

        #initial state is off
        self.state = 0
        self.first = first

        #Randomly initializing the weights
        for i in range(n_weights):
            weights.append(random.randint(0,10000)/10000)

    def next_state(activation, input = []):
        
        z = 0

        for i in len(weights):
            
            if(self.first):
                z += input[i] * weights[i]
            else:
                z += inputs[i].state * weights[i]
        

        #Update state based on inputs, weights and the activation function
        self.state = activation(z)

        return self.state

    