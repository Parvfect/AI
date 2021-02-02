import numpy as np
import matplotlib.pyplot as plt
import sys

inputs = np.array(np.random.rand(10))
weights = np.array(np.random.rand(10))
bias = 3

output = np.dot(inputs, weights) + bias

print(output)