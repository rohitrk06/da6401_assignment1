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
            self.model.weights[i] -= self.learning_rate * np.sum(weight_gradients[i],axis=0)
            self.model.biases[i] -= self.learning_rate * np.sum(bias_gradients[i])      

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


    
        
