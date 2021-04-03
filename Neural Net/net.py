import numpy as np 

# Loss function - calculate mean squared error
def mean_squared_error(actual, predicted):
    sum_square_error = 0.0
    for i in range(len(actual)):
        sum_square_error += (actual[i] - predicted[i])**2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    return mean_square

def Sigmoid(z):
    return 1/(1 + np.exp(-z))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs, activation):
        self.inputs = inputs
        self.output = activation.forward(np.dot(inputs, self.weights) + self.biases)
        self.output = [Sigmoid(i) for i in self.output]
        
#Activation functions
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

    def __init__(self, n_layers, input, output, hidden_layer_size, activation, learning_rate):
        
        self.n_layers = n_layers
        self.network = [Layer_Dense(input, hidden_layer_size)]
        for _ in range(n_layers - 2):
            self.network.append(Layer_Dense(hidden_layer_size, hidden_layer_size))
        
        self.network.append(Layer_Dense(hidden_layer_size, output))

        self.activation = activation
        self.learning_rate = learning_rate
        self.predicted = []
        self.actual = []
        self.loss = []

    def forward(self, inputs, outputs):
        
        self.actual = outputs
        for i in range(len(inputs)):
            self.network[0].forward(inputs[i], self.activation)
            for j in range(1,4):
                self.network[j].forward(self.network[j-1].output, self.activation)

            self.predicted.append(self.network[-1].output)
            n = len(self.predicted)
            self.loss.append((1/n) * (np.subtract(self.actual[:n], self.predicted)))
            self.back_propogate()
           
    def back_propogate(self):

        dCy = self.dCy()

        for i in range(3,0):
            layer = self.network[i]

            for i in range(0, len(layer.output)):
                dYz = self.dYz(layer.output[i])
                dZw = self.dZw(layer.inputs[i])
                dCw = dCy * dYz * dZw
                layer.weights[i] -= self.learning_rate * dCw
                layers.biases[i] -= self.learning_rate * dCy * dYz

    def dCy(self):
        
        n = len(self.predicted)
        outputs = self.actual[:len(self.predicted)]
        dCy = (2/n)(np.sum(outputs - self.predicted)) 
         
    def dYz(self, z):
        return Sigmoid(z) * (1 - Sigmoid(z))

    def dZw(self, x):
        return x


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


t = Network(4, 2, 1, 4, activation2, 0.05)
input = np.random.rand(1000,2)
output = np.random.rand(1000,1)
t.forward(input, output)
