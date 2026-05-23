import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.weights = np.random.randn()
        self.biases = 0.0

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.inputs * self.weights + self.biases

    def backward(self, dvalues):
        self.dweights = np.sum(dvalues * self.inputs)
        self.dbiasses = np.sum(dvalues)

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
        self.dloss = (-2 * self.y_input + 2 * self.y_prediction) / len(self.y_input)



# ================ DATA ==========
x = np.array([x for x in range(1, 100)])
y = 2 * x

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

prediction = []

# ============= MODEL =============
perceptron = Perceptron()
mse = MSE()

epochs = 100 # Sicne the non-compleity of this problem, 100 will be enough.
learning_rate = 0.0001
ax_x = 0
ax_y = 0

# ================= LEARNING PHASE ===============
for epoch in range(epochs):
    if epoch <= 10 and epoch % 2 == 0:
        perceptron.forward(x)

        ax[ax_y, ax_x].plot(
            x, y,
            color="red",
            label="Train Data"
        )

        ax[ax_y, ax_x].scatter(
            x, perceptron.outputs,
            color="blue",
            label="Prediction"
        )
        ax[ax_y, ax_x].legend()
        ax_x += 1
        if ax_x == 3:
            ax_x = 0
            ax_y += 1
    
    perceptron.forward(x)
    loss = mse.forward(y, perceptron.outputs)

    mse.backward()
    perceptron.backward(mse.dloss)

    perceptron.weights -= learning_rate * perceptron.dweights
    perceptron.biases -= learning_rate * perceptron.dbiasses



plt.show()


