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
-->For learning rate 0.01 and threshold 10

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

-->For learning rate 1 and threshold 10

Weights before training [0. 0. 0.]
Weights after 1 iteration [-1.  0.  0.]
Weights after 2 iteration [-1.  0.  1.]
Weights after 3 iteration [-2.  0.  1.]
Weights after 4 iteration [-2.  1.  1.]
Weights after 5 iteration [-2.  1.  2.]
Weights after 6 iteration [-2.  1.  2.]
Weights after 7 iteration [-2.  1.  2.]
Weights after 8 iteration [-2.  1.  2.]
Weights after 9 iteration [-2.  1.  2.]
Weights after 10 iteration [-2.  1.  2.]
For input [1 0] prediction is 0

-->For learning rate 10 and threshold 10

Weights before training [0. 0. 0.]
Weights after 1 iteration [-10.   0.   0.]
Weights after 2 iteration [-10.   0.  10.]
Weights after 3 iteration [-20.   0.  10.]
Weights after 4 iteration [-20.  10.  10.]
Weights after 5 iteration [-20.  10.  20.]
Weights after 6 iteration [-20.  10.  20.]
Weights after 7 iteration [-20.  10.  20.]
Weights after 8 iteration [-20.  10.  20.]
Weights after 9 iteration [-20.  10.  20.]
Weights after 10 iteration [-20.  10.  20.]
For input [1 0] prediction is 0

Conclusion:
    
    From the above examples we can see that we get corrrct weights values at iteration number 4 regardless of learning rate value.
    
'''