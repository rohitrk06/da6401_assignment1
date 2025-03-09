import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(x , derivative=False):
        '''
        Sigmoid activation function
        '''
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def relu(x, derivative=False):
        '''
        ReLU activation function
        '''
        if derivative:
            return np.where(x <= 0, 0, 1)
        
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x):
        '''
        Tanh activation function
        '''
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        '''
        Softmax activation function
        '''
        assert x.ndim == 2, "Input tensor should be a 2D tensor"
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
