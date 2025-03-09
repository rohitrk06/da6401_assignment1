import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(x , derivative=False):
        '''
        Sigmoid activation function
        '''
        sig = 1 / (1 + np.exp(-x))
        if derivative:
            return sig * (1 - sig)
        return sig
        
    @staticmethod
    def relu(x, derivative=False):
        '''
        ReLU activation function
        '''
        if derivative:
            return np.where(x <= 0, 0, 1)
        
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x, derivative=False):
        '''
        Tanh activation function and its derivative.
        '''
        tanh_x = np.tanh(x)  # Compute tanh
        if derivative:
            return 1 - tanh_x**2  # Compute derivative
        return tanh_x
    
    @staticmethod
    def softmax(x):
        '''
        Softmax activation function
        '''
        assert x.ndim == 2, "Input tensor should be a 2D tensor"
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
