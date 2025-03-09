import numpy as np


class NeuralNetwork():
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, activation_function, output_activation,loss_function = 'cross-entropy'):

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
        self.loss_function = loss_function

        self.forward_pass = []
        self.backward_pass = []


        self.weights = None
        self.biases = None

        self.__random_initialize_weights_and_biases()


    def __random_initialize_weights_and_biases(self): 
        self.weights = [np.random.randn(self.input_size, self.neurons_per_layer[0])*0.1] # 0.1 is multiplied to scale the weights down
        self.biases = [np.random.randn(self.neurons_per_layer[0])]

        for i in range(1, self.hidden_layers):
            self.weights.append(np.random.randn(self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
            self.biases.append(np.random.randn(self.neurons_per_layer[i]))

        self.weights.append(np.random.randn(self.neurons_per_layer[-1], self.output_size))
        self.biases.append(np.random.randn(self.output_size))

        # print("Weights and biases initialized successfully")
        # print("Dimensions of weights: ", [w.shape for w in self.weights])
        # print("Dimensions of biases: ", [b.shape for b in self.biases])

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        if self.loss_function == 'cross-entropy':
            loss = -np.sum(y*np.log(y_pred + 1e-8),axis=1).mean() ## Added 1e-8 to avoid log(0)
        else:
            raise NotImplementedError("Only cross-entropy loss is supported for now")
        return loss
    
    def compute_accuracy(self, X, y):
        y_pred = self.forward(X)
        return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))/y.shape[0]
 
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

        assert isinstance(y, np.ndarray), "Expected numpy array as input"
        assert y.ndim == 2, "Expected 2D numpy array as input"
        assert y.shape[1] == self.output_size, f"Output size mismatch. Expected {self.output_size}, got {y.shape[1]}"

        backward_pass_weights = []
        backward_pass_biases = []

        if self.loss_function == 'cross-entropy':
            gradient_wrt_pre_activation = -(y - self.forward_pass[-1][0])  # (64,10)
        else:
            raise NotImplementedError("Only cross-entropy loss is supported for now")

        for i in range(self.hidden_layers, 0, -1):
            # Compute weight gradients using einsum
            gradient_wrt_weights = np.einsum('bi,bj->bij', self.forward_pass[i-1][0], gradient_wrt_pre_activation)
            gradient_wrt_biases = np.sum(gradient_wrt_pre_activation, axis=0)

            # Compute gradient w.r.t. activation of previous layer
            gradient_wrt_activation = gradient_wrt_pre_activation @ self.weights[i].T
            gradient_wrt_pre_activation = gradient_wrt_activation * self.activation_function(self.forward_pass[i-1][-1], derivative=True)

            backward_pass_weights.append(gradient_wrt_weights.copy())
            backward_pass_biases.append(gradient_wrt_biases.copy())

        # Compute gradients for the first layer (input layer)
        gradient_wrt_weights = np.einsum('bi,bj->bij', X, gradient_wrt_pre_activation)
        gradient_wrt_biases = np.sum(gradient_wrt_pre_activation, axis=0)

        backward_pass_weights.append(gradient_wrt_weights.copy())
        backward_pass_biases.append(gradient_wrt_biases.copy())

        self.flag = False
        return (backward_pass_weights[::-1], backward_pass_biases[::-1])

    
            

