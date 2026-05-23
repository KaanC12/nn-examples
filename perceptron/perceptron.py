import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.weights = 0.01 * np.random.randn(1)
        self.biases = 0.0

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.inputs * self.weights + self.biases

    def backward(self, dvalues):
        self.dweights = dvalues * self.weights
        self.dbiasses = dvalues

class MSE:
    def forward(self, y_input, y_prediction):
        self.y_input = y_input
        self.y_prediction = y_prediction

        self.loss = np.mean((y_input - y_prediction) ** 2)

        return self.loss

    """
    Sicne we want to update the prediction, we should find the derivation 
    according to y'
    (y - y')^2 : d/d'y (y - y')^2 = -2y + 2y'  
    """
    def backward(self):
        self.dloss = -2 * self.y_input + 2 * self.y_prediction




x = np.array([x for x in range(1, 100)])
y = 2 * x


# ============= MODEL =============
perceptron = Perceptron()
mse = MSE()

epochs = 100 # Sicne the non-compleity of this problem, 100 will be enough.
learning_rate = 0.01

for epoch in range(epochs):
    perceptron.forward(x)
    loss = mse.forward(perceptron.output, y)

    mse.backward()
    perceptron.backward(mse.dloss)

    perceptron.weights -= learning_rate * perceptron.dweights
    perceptron.biases -= learning_rate * perceptron.dbiasses



