import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import loss as loss

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs, activation):
        self.output = activation.forward(np.dot(inputs, self.weights) + self.biases)
        

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


class Network:

    def __init__(self, n_layers, input, output, hidden_layer_size, activation):
        
        self.n_layers = n_layers
        self.network = [Layer_Dense(input, hidden_layer_size)]
        for _ in range(n_layers - 2):
            self.network.append(Layer_Dense(hidden_layer_size, hidden_layer_size))
        
        self.network.append(Layer_Dense(hidden_layer_size, output))

        self.activation = activation

    def forward(self, inputs, outputs):
        
        for i in range(len(inputs)):
            self.network[0].forward(inputs[i], self.activation)
            for j in range(1,4):
                self.network[j].forward(self.network[j-1].output, self.activation)

            self.result = self.network[-1].output
            print(self.result - output[i])
        
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

"""
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

"""


t = Network(4, 2, 2, 4, activation2)
input = np.random.rand(1000,2)
output = np.random.rand(1000, 2)
t.forward(input, output)

