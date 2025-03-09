import numpy as np


class NeuralNetwork():
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, activation_function, output_activation):

        assert hidden_layers > 0, "Number of hidden layers should be greater than 0"
        if type(neurons_per_layer) == int:
            neurons_per_layer = [neurons_per_layer]*hidden_layers
        assert len(neurons_per_layer) == hidden_layers, "Length of neurons_per_layer list should be equal to the number of hidden layers"
        
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.output_activation = output_activation

        self.weights = None
        self.biases = None

        self.__initialize_weights_and_biases()

    def __initialize_weights_and_biases(self): 
        self.weights = [np.random.randn(self.input_size, self.neurons_per_layer[0])]
        self.biases = [np.random.randn(self.neurons_per_layer[0])]

        for i in range(1, self.hidden_layers):
            self.weights.append(np.random.randn(self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
            self.biases.append(np.random.randn(self.neurons_per_layer[i]))

        self.weights.append(np.random.randn(self.neurons_per_layer[-1], self.output_size))
        self.biases.append(np.random.randn(self.output_size))

    def forward(self, X):
        assert X.shape[1] == self.input_size, f"Input size is not matching with the input size of the network. Expected {self.input_size} but got {X.shape[1]}"
        f_x = X@self.weights[0] + self.biases[0]
        h_x = self.activation_function(f_x)

        for i in range(1, self.hidden_layers):
            f_x = h_x @ self.weights[i] + self.biases[i]
            h_x = self.activation_function(f_x)

        f_x = h_x @ self.weights[-1] + self.biases[-1]
        y = self.output_activation(f_x)

        return y

    def backward(self, x, y):
        pass 


         