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

        self.forward_pass = []
        self.backward_pass = []


        self.weights = None
        self.biases = None

        self.__random_initialize_weights_and_biases()


    def __random_initialize_weights_and_biases(self): 
        self.weights = [np.random.randn(self.input_size, self.neurons_per_layer[0])]
        self.biases = [np.random.randn(self.neurons_per_layer[0])]

        for i in range(1, self.hidden_layers):
            self.weights.append(np.random.randn(self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
            self.biases.append(np.random.randn(self.neurons_per_layer[i]))

        self.weights.append(np.random.randn(self.neurons_per_layer[-1], self.output_size))
        self.biases.append(np.random.randn(self.output_size))
 
    def forward(self, X):
        self.flag = True
        assert X.shape[1] == self.input_size, f"Input size is not matching with the input size of the network. Expected {self.input_size} but got {X.shape[1]}"
        f_x = X@self.weights[0] + self.biases[0]
        h_x = self.activation_function(f_x)

        self.forward_pass.append((h_x.copy(), f_x.copy()))

        for i in range(1, self.hidden_layers):
            f_x = h_x @ self.weights[i] + self.biases[i]
            h_x = self.activation_function(f_x)
            self.forward_pass.append((h_x.copy(), f_x.copy()))

        f_x = h_x @ self.weights[-1] + self.biases[-1]
        y = self.output_activation(f_x)
        self.forward_pass.append((y.copy(), f_x.copy()))

        return y

    def backward(self, X, y):
        assert self.flag == True, "Forward pass should be called before calling backward pass"
        
        assert type(y) == np.ndarray, "Expected numpy array as input"
        assert y.ndim == 2, "Expected 2D numpy array as input"
        assert y.shape[1] == self.output_size, f"Output size is not matching with the output size of the network. Expected {self.output_size} but got {y.shape[1]}"

        gradient_wrt_pre_activation = -(y - self.forward_pass[-1][0]) # gradient of cross entropy loss

        for i in range(self.hidden_layers, 0, -1):
            gradient_wrt_weights = gradient_wrt_pre_activation[:, :, np.newaxis] @ self.forward_pass[i-1][0][:, np.newaxis, :]
            gradient_wrt_biases = gradient_wrt_pre_activation
            gradient_wrt_activation = gradient_wrt_pre_activation @ self.weights[i].T
            gradient_wrt_pre_activation = gradient_wrt_activation * self.activation_function(self.forward_pass[i-1][-1], derivative=True)
            self.backward_pass.append((gradient_wrt_weights, gradient_wrt_biases))

        gradient_wrt_weights = gradient_wrt_pre_activation[:, :, np.newaxis] @ X[:, np.newaxis, :]
        gradient_wrt_biases = gradient_wrt_pre_activation

        self.backward_pass.append((gradient_wrt_weights, gradient_wrt_biases))

        return self.backward_pass

