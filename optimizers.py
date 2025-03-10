import numpy as np

class SGD:
    def __init__(self, learning_rate,epochs, batch_size, model, weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.weight_decay = weight_decay

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            # print(np.sum(weight_gradients[i],axis=0))
            self.model.weights[i] -= self.learning_rate * np.mean(weight_gradients[i],axis=0) - self.weight_decay * self.model.weights[i]
            self.model.biases[i] -= self.learning_rate * np.mean(bias_gradients[i]) - self.weight_decay * self.model.biases[i] 

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
    def __init__(self, learning_rate, epochs, batch_size,model, momentum=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] +  self.learning_rate * np.mean(weight_gradients[i],axis=0)
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * np.mean(bias_gradients[i]) 

            self.model.weights[i] -=  self.velocity_weights[i] - self.weight_decay * self.model.weights[i]
            self.model.biases[i] -=  self.velocity_biases[i] - self.weight_decay * self.model.biases[i]

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
    def __init__(self, learning_rate, epochs, batch_size, model, momentum=0.9,weight_decay = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):        
        weights_before_lookahead = [w.copy() for w in self.model.weights]
        biases_before_lookahead = [b.copy() for b in self.model.biases]
        self.model.forward(X)
        

        self.model.weights = [ weights - self.momentum*velocity for weights, velocity in zip(self.model.weights, self.velocity_weights)]
        self.model.biases = [ biases - self.momentum*velocity for biases, velocity in zip(self.model.biases, self.velocity_biases)]

        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * np.mean(weight_gradients[i],axis=0) 
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * np.mean(bias_gradients[i]) 

            self.model.weights[i] = weights_before_lookahead[i] -  self.velocity_weights[i] - self.weight_decay * self.model.weights[i]
            self.model.biases[i] = biases_before_lookahead[i] -  self.velocity_biases[i] - self.weight_decay * self.model.biases[i]

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

class RMSProp:
    def __init__(self, learning_rate, epochs, batch_size, model, beta=0.9, epsilon=1e-8,weight_decay=0):
        self.inital_learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.accumulated_weight_gradients = [np.zeros_like(w) for w in self.model.weights]
        self.accumulated_bias_gradients = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.accumulated_weight_gradients[i] = self.beta * self.accumulated_weight_gradients[i] + (1-self.beta) * np.mean(weight_gradients[i]**2,axis=0)
            self.accumulated_bias_gradients[i] = self.beta * self.accumulated_bias_gradients[i] + (1-self.beta) * np.mean(bias_gradients[i]**2)

            # print(np.sqrt(self.accumulated_weight_gradients[i]).shape)

            self.model.weights[i] -= self.inital_learning_rate * np.mean(weight_gradients[i]**2,axis=0) / (np.sqrt(self.accumulated_weight_gradients[i]) + self.epsilon) - self.weight_decay * self.model.weights[i]
            self.model.biases[i] -= self.inital_learning_rate * np.mean(bias_gradients[i]**2) / (np.sqrt(self.accumulated_bias_gradients[i]) + self.epsilon) - self.weight_decay * self.model.biases[i]

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


class Adam:
    def __init__(self, learning_rate, epochs, batch_size, model, beta1=0.9, beta2=0.999, epsilon=1e-8 , weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay =weight_decay
        self.moment_weights = [np.zeros_like(w) for w in self.model.weights]
        self.moment_biases = [np.zeros_like(b) for b in self.model.biases]
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.moment_weights[i] = self.beta1 * self.moment_weights[i] + (1-self.beta1) * np.mean(weight_gradients[i],axis=0)
            self.moment_biases[i] = self.beta1 * self.moment_biases[i] + (1-self.beta1) * np.mean(bias_gradients[i])

            self.velocity_weights[i] = self.beta2 * self.velocity_weights[i] + (1-self.beta2) * np.mean(weight_gradients[i]**2,axis=0)
            self.velocity_biases[i] = self.beta2 * self.velocity_biases[i] + (1-self.beta2) * np.mean(bias_gradients[i]**2)

            moment_weights_hat = self.moment_weights[i] / (1 - self.beta1**(i+1))
            moment_biases_hat = self.moment_biases[i] / (1 - self.beta1**(i+1))

            velocity_weights_hat = self.velocity_weights[i] / (1 - self.beta2**(i+1))
            velocity_biases_hat = self.velocity_biases[i] / (1 - self.beta2**(i+1))

            self.model.weights[i] -= self.learning_rate * moment_weights_hat / (np.sqrt(velocity_weights_hat) + self.epsilon) - self.weight_decay * self.model.weights[i]
            self.model.biases[i] -= self.learning_rate * moment_biases_hat / (np.sqrt(velocity_biases_hat) + self.epsilon) - self.weight_decay * self.model.biases[i]

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

class NADAM:
    def __init__(self, learning_rate, epochs, batch_size, model, beta1=0.9, beta2=0.999, epsilon=1e-8,weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.moment_weights = [np.zeros_like(w) for w in self.model.weights]
        self.moment_biases = [np.zeros_like(b) for b in self.model.biases]
        self.velocity_weights = [np.zeros_like(w) for w in self.model.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.model.biases]

    def __update_weights(self, X, y):
        self.model.forward(X)
        weight_gradients, bias_gradients = self.model.backward(X,y)

        for i in range(len(self.model.weights)):
            self.moment_weights[i] = self.beta1 * self.moment_weights[i] + (1-self.beta1) * np.mean(weight_gradients[i],axis=0)
            self.moment_biases[i] = self.beta1 * self.moment_biases[i] + (1-self.beta1) * np.mean(bias_gradients[i])

            self.velocity_weights[i] = self.beta2 * self.velocity_weights[i] + (1-self.beta2) * np.mean(weight_gradients[i]**2,axis=0)
            self.velocity_biases[i] = self.beta2 * self.velocity_biases[i] + (1-self.beta2) * np.mean(bias_gradients[i]**2)

            moment_weights_hat = self.moment_weights[i] / (1 - self.beta1**(i+1))
            moment_biases_hat = self.moment_biases[i] / (1 - self.beta1**(i+1))

            velocity_weights_hat = self.velocity_weights[i] / (1 - self.beta2**(i+1))
            velocity_biases_hat = self.velocity_biases[i] / (1 - self.beta2**(i+1))

            self.model.weights[i] -= self.learning_rate * (self.beta1*moment_weights_hat + (1-self.beta1)*np.mean(weight_gradients[i],axis=0)/(1-self.beta1**(i+1))) / (np.sqrt(velocity_weights_hat) + self.epsilon) - self.weight_decay * self.model.weights[i]
            self.model.biases[i] -= self.learning_rate * (self.beta1*moment_biases_hat + (1-self.beta1)*np.mean(bias_gradients[i])/(1-self.beta1**(i+1))) / (np.sqrt(velocity_biases_hat) + self.epsilon) - self.weight_decay * self.model.biases[i]

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