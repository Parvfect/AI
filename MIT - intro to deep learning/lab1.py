"""Lab 1 a for mit intro to deep learning"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

#Tensorflow handles the flow of tensors - n dimensional vectors of base data type

#Defining 0 dimensional tensors
sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

#print("'sport' is a {}-d Tensor".format(tf.rank(sport).numpy()))
#print("'number' is a {}-d Tensor".format(tf.rank(number).numpy()))

#Defining 1 dimensional tensors
sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([2.3,3.4], tf.float64)

#print("'sports' is a {}-d Tensor".format(tf.rank(sports).numpy()))
#print("'numbers' is a {}-d Tensor".format(tf.rank(numbers).numpy()))

class OurDenseLayer(tf.keras.layers.Layer):
    
    def __init__(self, n_ouptut_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_ouptut_nodes
    
    def build(self, input_shape):
        d = int(input_shape[-1])
        #Define and initialize parameters weights and bias
        #Parameter initialization is random
        self.W = self.add_weight("weight", shape = [d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape = [1, self.n_output_nodes])

    def call(self, x):
        z = tf.matmul(x, self.W) + self.b
        y = tf.math.sigmoid(z)
        return y

tf.random.set_seed(1)
layer = OurDenseLayer(1)
layer.build((1,2))
x_input = tf.constant([[1,2.]], shape= (1,2))
y = layer.call(x_input)

#print(y.numpy())

##Defining a neural net using the Keras API

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense

#Defining a network using the sequential API

n_ouput_nodes = 3 

model = Sequential()

dense_layer = Dense(n_ouput_nodes, activation='sigmoid')
model.add(dense_layer)

x_input = tf.constant([[1,2.]], shape = (1,2))
y = layer.call(x_input)

#print(y)
#print(y.numpy())

#Defining a network using subclassing

class SubClassModel(tf.keras.Model):

    def __init__(self, n_ouptut_nodes):
        super(SubClassModel, self).__init__()
        self.dense_layer = Dense(n_ouptut_nodes, activation='sigmoid')

    def call(self, inputs):
        return self.dense_layer(inputs)


model = SubClassModel(3)
print(model.call(x_input))

class IdentityModel(tf.keras.Model):

    def __init__(self , n_ouptut_nodes, is_identity = True):
        super(IdentityModel, self). __init__()
        self.is_identity = True
        self.dense_layer = Dense(n_ouptut_nodes, activation = 'relu')

    def call(self, inputs):
        if self.is_identity:
            return inputs
        else:
            return self.dense_layer(inputs)

x = tf.Variable(3.0)

#Initiate the gradient tape
with tf.GradientTape() as tape:
    #Define the function
    y = x*x

derivative = tape.gradient(y, x)

assert derivative.numpy() == 6.0

#Function minimisation with SGD and automatic diffrentiation#

x = tf.Variable([tf.random.normal([1])])
print("Initializing x as {}".format(x))

learning_rate = 1e-2
history = []
 
#Define the target value
x_f = 4
 
for i in range(500):
    with tf.GradientTape() as g:
        loss = (x - x_f)**2
    grad = g.gradient(loss, x)
    new_x = x - learning_rate*grad
    x.assign(new_x)
    history.append(x.numpy()[0])

plt.plot(history)
plt.plot([0,500], [x_f, x_f])
plt.legend("Predicted", "True")
plt.xlabel("Iteration")
plt.ylabel("x value ")
plt.show()