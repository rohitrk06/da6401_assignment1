import numpy as np

class SGD:
    def __init__(self, learning_rate,epochs, batch_size, model):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            # print(np.sum(weight_gradients[i],axis=0))
            self.model.weights[i] -= self.learning_rate * np.mean(weight_gradients[i],axis=0)
            self.model.biases[i] -= self.learning_rate * np.mean(bias_gradients[i])      

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                self.__update_weights(X_batch, y_batch)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Loss: {self.model.compute_loss(X, y)}")
            print("Accuracy: ", self.model.compute_accuracy(X, y))
            print("========================================")

class Momentum:
    def __init__(self, learning_rate, epochs, batch_size, model, momentum=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.momentum = momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] +  np.mean(weight_gradients[i],axis=0)
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + np.mean(bias_gradients[i])

            self.model.weights[i] -= self.learning_rate * self.velocity_weights[i]
            self.model.biases[i] -= self.learning_rate * self.velocity_biases[i]

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                self.__update_weights(X_batch, y_batch)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Loss: {self.model.compute_loss(X, y)}")
            print("Accuracy: ", self.model.compute_accuracy(X, y))
            print("========================================")

class NestrovAcceleratedGradient:
    def __init__(self, learning_rate, epochs, batch_size, model, momentum=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.momentum = momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):        
        weights_before_lookahead = [w.copy() for w in self.model.weights]
        biases_before_lookahead = [b.copy() for b in self.model.biases]


        self.model.weights = [ weights - self.momentum*velocity for weights, velocity in zip(self.model.weights, self.velocity_weights)]
        self.model.biases = [ biases - self.momentum*velocity for biases, velocity in zip(self.model.biases, self.velocity_biases)]

        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * np.mean(weight_gradients[i],axis=0)
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * np.mean(bias_gradients[i])

            self.model.weights[i] = weights_before_lookahead[i] -  self.velocity_weights[i]
            self.model.biases[i] = biases_before_lookahead[i] -  self.velocity_biases[i]

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                self.__update_weights(X_batch, y_batch)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Loss: {self.model.compute_loss(X, y)}")
            print("Accuracy: ", self.model.compute_accuracy(X, y))
            print("========================================")