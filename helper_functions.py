import numpy as np

def train_val_split(x_train, y_train, val_size = 0.2):
    np.random.seed(42)
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    # print("y_train shape: ", y_train.shape)
    # print("x_train shape: ", x_train.shape)

    x_val = x_train[:int(val_size*x_train.shape[0])]
    y_val = y_train[:int(val_size*x_train.shape[0])]

    training_x = x_train[int(val_size*x_train.shape[0]):]
    training_y = y_train[int(val_size*x_train.shape[0]):]

    # print("Training data shape: ", x_train.shape, y_train.shape)
    # print("Validation data shape: ", x_val.shape, y_val.shape)

    return training_x, training_y, x_val, y_val

def normalise_and_flatten(data):
    data = data.reshape(data.shape[0], -1)
    return data/255

def one_hot_encoding(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

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
