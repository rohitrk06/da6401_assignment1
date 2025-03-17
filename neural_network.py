import numpy as np
from helper_functions import ActivationFunctions
from optimizers import *

class NeuralNetwork:
    def __init__(self, input_size, output_size,
                 num_hidden_layer, hidden_layer_size,
                 weight_init_startegy = "random",
                 activation_function = ActivationFunctions.sigmoid, output_activation=ActivationFunctions.softmax,
                 loss_function="cross_entropy"):
        
        self.input_size = input_size
        self.output_size = output_size

        self.num_hidden_layer = num_hidden_layer

        # If hidden_layer_size is an integer, then all hidden layers will have the same number of neurons
        # If hidden_layer_size is a list, assuming that the length of the list is equal to num_hidden_layer,
        # each element of the list will be the number of neurons in the corresponding hidden layer
        if isinstance(hidden_layer_size, int):
            self.hidden_layer_size = [hidden_layer_size] * num_hidden_layer
        else:
            assert len(hidden_layer_size) == num_hidden_layer, "Length of hidden_layer_size should be equal to num_hidden_layer"
            self.hidden_layer_size = hidden_layer_size

        self.weights = []
        self.biases = []

        #Initialize weights and biases according to the passed argument
        if weight_init_startegy == "random":
            self.__init_random_weights()
        elif weight_init_startegy == "xavier":
            self.__init_xavier_weights()
        else:
            raise ValueError("Invalid weight initialization strategy")
        
        self.activation_fn = activation_function
        self.output_activation = output_activation
        self.loss_fn = loss_function
        # optimizer = optimizer

        self.forward_pass_values = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, X):
        assert X.shape[-1] == self.input_size, f"Input size is not matching with the input size of the network. Expected {self.input_size} but got {X.shape[1]}"
        self.forward_pass_values = []
        pre_activation = X@self.weights[0] + self.biases[0]
        post_activation = self.activation_fn(pre_activation)

        self.forward_pass_values.append((pre_activation.copy(),post_activation.copy()))

        for i in range(1, self.num_hidden_layer):
            pre_activation = post_activation@self.weights[i] + self.biases[i]
            post_activation = self.activation_fn(pre_activation)

            self.forward_pass_values.append((pre_activation.copy(),post_activation.copy()))
        
        pre_activation = post_activation@self.weights[self.num_hidden_layer] + self.biases[self.num_hidden_layer]
        output = self.output_activation(pre_activation)
        self.forward_pass_values.append((pre_activation.copy(),output.copy()))
        return output

    def backward(self, X, y):

        assert isinstance(y, np.ndarray), "Expected numpy array as input"
        assert y.ndim == 2, "Expected 2D numpy array as input"
        assert y.shape[1] == self.output_size, f"Output size mismatch. Expected {self.output_size}, got {y.shape[1]}"
        assert X.shape[-1] == self.input_size, f"Input size is not matching with the input size of the network. Expected {self.input_size} but got {X.shape[1]}"

        self.weight_gradients = []
        self.bias_gradients = []

        if self.loss_fn == "cross_entropy":
            gradient_wrt_preactivation_layer = -(y-self.forward_pass_values[-1][1])
        elif self.loss_fn == "mean_squared_error":
            softmax_output = self.forward_pass_values[-1][1]  # Softmax output (y_pred)
            batch_size = y.shape[0]
            
            # Compute the Jacobian matrix for softmax
            jacobian = np.zeros((batch_size, self.output_size, self.output_size))
            for i in range(batch_size):
                s = softmax_output[i].reshape(-1, 1)  # Convert to column vector
                jacobian[i] = np.diagflat(s) - np.dot(s, s.T)  # Compute Softmax' Jacobian

            # Compute gradient using the Jacobian
            gradient_wrt_preactivation_layer = np.einsum("bij,bj->bi", jacobian, (softmax_output - y)) * (-2 / batch_size)
        else:
            raise ValueError("Invalid values for Loss function. Allowed values: (\"cross_entropy\", \"mse\")")
        
        for layer in range(self.num_hidden_layer,0,-1):
            gradient_wrt_weights = np.sum(np.einsum('bi,bj->bji',gradient_wrt_preactivation_layer,self.forward_pass_values[layer-1][1]),axis=0)
            gradient_wrt_biases = np.sum(gradient_wrt_preactivation_layer)

            self.weight_gradients.append(gradient_wrt_weights.copy())
            self.bias_gradients.append(gradient_wrt_biases.copy())

            gradient_wrt_activation_layer = np.einsum("ji,bi->bj ",self.weights[layer],gradient_wrt_preactivation_layer)
            gradient_wrt_preactivation_layer = gradient_wrt_activation_layer * self.activation_fn(self.forward_pass_values[layer-1][0],derivative=True)

        gradient_wrt_weights = np.sum(np.einsum('bi,bj->bji',gradient_wrt_preactivation_layer,X),axis=0)
        gradient_wrt_biases = np.sum(gradient_wrt_preactivation_layer)

        self.weight_gradients.append(gradient_wrt_weights.copy())
        self.bias_gradients.append(gradient_wrt_biases.copy())

        return self.weight_gradients[::-1],self.bias_gradients[::-1]


    def __init_random_weights(self):
        self.weights.append((np.random.randn(self.input_size, self.hidden_layer_size[0])) * 0.1)
        self.biases.append(np.random.randn(1,self.hidden_layer_size[0])*0.1)

        for i_layer in range(1,self.num_hidden_layer):
            self.weights.append(np.random.randn(self.hidden_layer_size[i_layer - 1], self.hidden_layer_size[i_layer]) * 0.1)
            self.biases.append(np.random.randn(1,self.hidden_layer_size[i_layer])*0.1)

        self.weights.append(np.random.randn(self.hidden_layer_size[-1],self.output_size)*0.1)
        self.biases.append(np.random.randn(1,self.output_size)*0.1)
        
    def __init_xavier_weights(self,type="normal-xavier"): 
        if type=="normal-xavier":
            std = np.sqrt(2/(self.input_size + self.hidden_layer_size[0]))
            self.weights.append(np.random.normal(0,std,size=(self.input_size, self.hidden_layer_size[0])))
            self.biases.append(np.random.normal(0,std,size=(1,self.hidden_layer_size[0])))

            for i_layer in range(1,self.num_hidden_layer):
                std = np.sqrt(2/(self.hidden_layer_size[i_layer-1] + self.hidden_layer_size[i_layer]))
                self.weights.append(np.random.normal(0,std,size=(self.hidden_layer_size[i_layer-1], self.hidden_layer_size[i_layer])))
                self.biases.append(np.random.normal(0,std,size=(1,self.hidden_layer_size[i_layer])))
            
            std = np.sqrt(2/(self.hidden_layer_size[-1]+self.output_size))
            self.weights.append(np.random.normal(0,std,size=(self.hidden_layer_size[-1],self.output_size)))
            self.biases.append(np.random.normal(0,std,size=(1,self.output_size)))

        elif type=="uniform-xavier":
            limit = np.sqrt(6/(self.input_size + self.hidden_layer_size[0]))
            self.weights.append(np.random.uniform(-limit,limit,size=(self.input_size, self.hidden_layer_size[0])))
            self.biases.append(np.random.uniform(-limit,limit,size=(1,self.hidden_layer_size[0])))

            for i_layer in range(1,self.num_hidden_layer):
                limit = np.sqrt(6/(self.hidden_layer_size[i_layer-1] + self.hidden_layer_size[i_layer]))
                self.weights.append(np.random.uniform(-limit,limit,size=(self.hidden_layer_size[i_layer-1], self.hidden_layer_size[i_layer])))
                self.biases.append(np.random.uniform(-limit,limit,size=(1,self.hidden_layer_size[i_layer])))
            
            limit = np.sqrt(6/(self.hidden_layer_size[-1]+self.output_size))
            self.weights.append(np.random.uniform(-limit,limit,size=(self.hidden_layer_size[-1],self.output_size)))
            self.biases.append(np.random.uniform(-limit,limit,size=(1,self.output_size)))

        else:
            raise ValueError("Invalid Xavier Weight Initialization type. Allowed values (normal-xavier, uniform-xavier)")
        
    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        if self.loss_fn == 'cross_entropy':
            loss = -np.sum(y*np.log(y_pred + 1e-8),axis=1).mean() ## Added 1e-8 to avoid log(0)
        elif self.loss_fn == 'mean_squared_error':
            loss = np.sum((y-y_pred)**2)/y.shape[0]
        else:
            raise NotImplementedError("Only cross_entropy loss is supported for now")
        return loss
    
    def get_learning_params(self):
        return {"weights": self.weights, "biases": self.biases}
    
    def compute_accuracy(self, X, y):
        return np.sum(np.argmax(y, axis=1) == self.predict(X))/y.shape[0]
    
    def predict(self, X):
        return np.argmax(self.forward(X),axis=1)
    
    def train(self,optimizer,X,y,X_val=None,y_val=None):
        print("Training the model with optimizer: ", optimizer.__class__.__name__)
        for epoch in range(optimizer.epochs):
            print(f"Epoch: {epoch+1}/{optimizer.epochs}")
            for i in range(0,len(X),optimizer.batch_size):
                X_batch = X[i:i+optimizer.batch_size]
                y_batch = y[i:i+optimizer.batch_size]

                optimizer.update_weights(self,X_batch,y_batch)

            train_loss = self.compute_loss(X,y)
            train_accuracy = self.compute_accuracy(X,y)
            print(f"Loss: {train_loss}, Accuracy: {train_accuracy}")
            wandb.log({"train_accuracy": train_accuracy})
            wandb.log({"train_loss": train_loss})
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val,y_val)
                val_accuracy = self.compute_accuracy(X_val,y_val)
                print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
                wandb.log({"val_loss": val_loss})
                wandb.log({"val_accuracy": val_accuracy})
            
        
