from fashion_mnist import x_train, y_train
from neural_network import NeuralNetwork
import numpy as np
from helper_functions import *
from optimizers import SGD, Momentum, NestrovAcceleratedGradient, RMSProp

# Do the train and val split using numpy
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

# print("y_train shape: ", y_train.shape)
y_train = one_hot_encoding(y_train, 10)
# print(y_train.shape)

x_train = normalise_and_flatten(x_train)
x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
# print("Data loaded successfully")
# print("Training data shape: ", x_train.shape, y_train.shape)
# print("Validation data shape: ", x_val.shape, y_val.shape)


nn = NeuralNetwork(input_size = 784, output_size = 10, hidden_layers = 4, neurons_per_layer = 32, activation_function = ActivationFunctions.tanh, output_activation = ActivationFunctions.softmax, loss_function = 'cross-entropy')
# sgd_optimizer = SGD(0.1, 5, 64, nn)

# momuntum_optimizer = Momentum(0.001, 5, 64, nn, momentum=0.1)
# nag = NestrovAcceleratedGradient(0.001, 5, 64, nn, momentum=0.5)

rms_prop = RMSProp(0.0001, 5, 64, nn, beta=0.9, epsilon=1e-8)


print("Training the model")
# sgd_optimizer.train(x_train, y_train)

# momuntum_optimizer.train(x_train, y_train)

# nag.train(x_train, y_train)

rms_prop.train(x_train, y_train)

print("Training completed")
print("Loss on validation set: ", nn.compute_loss(x_val, y_val))
print("Accuracy on validation set: ", nn.compute_accuracy(x_val, y_val))
print("Accuracy on training set: ", nn.compute_accuracy(x_train, y_train))



