# Testing whether data is linear or not linear

'''
 There are 4 inputs in dataset of movie 
 1) Hollywood Movie
 2) Actor: Tom Cruise
 3) Genre: Action Movie
 4) IMDB > 8

Here all inputs are boolean and result will indicate that if i will like that movie or not

'''

import numpy as np 
from Perceptron import Perceptron

training_inputs = []

training_inputs.append(np.array([0,0,0,0]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([1,1,1,1]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([1,0,0,1]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([0,0,0,0]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([1,1,1,0]))


labels = np.array([0,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1])

perceptron  = Perceptron(4)
perceptron.train(training_inputs, labels)

inputs = np.array([1,1,1,0])

print("Actual Output : "+str(1))
print("Predicted output : "+str(perceptron.predict(inputs)))

print("=========================================================")

inputs = np.array([1,1,0,0])

print("Actual Output : "+str(0))
print("Predicted output : "+str(perceptron.predict(inputs)))

'''
Weights before training [0. 0. 0. 0. 0.]
Weights after 1 iteration [-0.01  0.01  0.    0.01  0.01]
Weights after 2 iteration [-0.01  0.01  0.01  0.02  0.02]
Weights after 3 iteration [-0.02  0.01  0.01  0.01  0.02]
Weights after 4 iteration [-0.03  0.02  0.01  0.01  0.02]
Weights after 5 iteration [-0.03  0.02  0.01  0.01  0.03]
Weights after 6 iteration [-0.03  0.02  0.02  0.02  0.03]
Weights after 7 iteration [-0.04  0.02  0.01  0.02  0.03]
Weights after 8 iteration [-0.04  0.01  0.02  0.02  0.03]
Weights after 9 iteration [-0.05  0.02  0.02  0.02  0.03]
Weights after 10 iteration [-0.05  0.02  0.02  0.02  0.04]
Weights after 11 iteration [-0.05  0.02  0.03  0.02  0.04]
Weights after 12 iteration [-0.06  0.02  0.02  0.03  0.04]
Weights after 13 iteration [-0.06  0.03  0.02  0.02  0.05]
Weights after 14 iteration [-0.06  0.02  0.02  0.03  0.05]
Weights after 15 iteration [-0.06  0.02  0.03  0.03  0.05]
Weights after 16 iteration [-0.06  0.03  0.03  0.04  0.05]
Weights after 17 iteration [-0.07  0.03  0.02  0.03  0.05]
Weights after 18 iteration [-0.07  0.02  0.03  0.03  0.05]
Weights after 19 iteration [-0.07  0.02  0.04  0.03  0.05]
Weights after 20 iteration [-0.07  0.02  0.03  0.04  0.05]
Weights after 21 iteration [-0.08  0.03  0.03  0.03  0.05]
Weights after 22 iteration [-0.08  0.03  0.03  0.03  0.06]
Weights after 23 iteration [-0.08  0.02  0.03  0.04  0.06]
Weights after 24 iteration [-0.08  0.02  0.04  0.04  0.06]
Weights after 25 iteration [-0.08  0.02  0.03  0.05  0.06]
Weights after 26 iteration [-0.09  0.03  0.03  0.04  0.06]
Weights after 27 iteration [-0.08  0.03  0.04  0.04  0.07]
Weights after 28 iteration [-0.08  0.03  0.04  0.04  0.07]
Weights after 29 iteration [-0.08  0.03  0.04  0.04  0.07]
Weights after 30 iteration [-0.08  0.03  0.04  0.04  0.07]
Actual Output : 1
Predicted output : 1
=========================================================
Actual Output : 0
Predicted output : 0


Conclusion :
    
Here we can see that our perceptron is predicting the correct values. Hence we can say that our data is linear
'''
