import numpy as np
from Perceptron import Perceptron

training_inputs=[]
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0])

perceptron = Perceptron(no_of_inputs=2, threshold=10)
perceptron.train(training_inputs, labels)

inputs = np.array([1,0])

print("For input "+str(inputs)+" prediction is "+str(perceptron.predict(inputs)))

'''
Weights before training [0. 0. 0.]
Weights after 1 iteration [-0.01  0.    0.  ]
Weights after 2 iteration [-0.01  0.    0.01]
Weights after 3 iteration [-0.02  0.    0.01]
Weights after 4 iteration [-0.02  0.01  0.01]
Weights after 5 iteration [-0.02  0.01  0.02]
Weights after 6 iteration [-0.02  0.01  0.02]
Weights after 7 iteration [-0.02  0.01  0.02]
Weights after 8 iteration [-0.02  0.01  0.02]
Weights after 9 iteration [-0.02  0.01  0.02]
Weights after 10 iteration [-0.02  0.01  0.02]
For input [1 0] prediction is 0

'''