from neural_network import NeuralNetwork
import numpy as np
from helper_functions import *
from timestampLogger import *
from optimizers import *
from fashion_mnist import x_train, y_train, x_test, y_test
import argparse
from keras.datasets import fashion_mnist, mnist
import wandb
import sys

# log_file = open("log.txt", "a")
# logger = TimeStampLogger(log_file)
# sys.stdout = logger


# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 
# x_train = normalise_and_flatten(x_train)
# y_train = one_hot_encoding(y_train, 10) # One hot encoding the labels

# x_test = normalise_and_flatten(x_test)
# # y_test = one_hot_encoding(y_test, 10) # One hot encoding the labels

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)

def get_args():
    parser = argparse.ArgumentParser(description="Neural Network Training Script with Weights & Biases")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment1", help="Project name used in WandB dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="rohitrk06-indian-institute-of-technology-madras", help="WandB Entity")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="xavier", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "relu"], default="tanh", help="Activation function")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    run = wandb.init(
        project = args.wandb_project,
        entity = args.wandb_entity,
        job_type="training"
    )

    wandb.run.name = f"hls_{args.num_layers}_hs_{args.hidden_size}_bs_{args.batch_size}_act_{args.activation}_opt_{args.optimizer}_lr_{args.learning_rate}_weight_{args.weight_init}_wd_{args.weight_decay}_loss_{args.loss}_epoch_{args.epochs}_id_test_data_{wandb.run.id}"

    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (x_train,y_train), (x_test,y_test) = mnist.load_data()
    else:
        raise ValueError("Invalid Dataset. Allowed Values (\"fashion_mnist\", \"mnist\")")


    x_train = normalise_and_flatten(x_train)
    y_train = one_hot_encoding(y_train, 10) # One hot encoding the labels

    x_test = normalise_and_flatten(x_test)
    # y_test = one_hot_encoding(y_test, 10) # One hot encoding the labels

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    if args.activation == "sigmoid":
        activation_function = ActivationFunctions.sigmoid
    elif args.activation == "ReLU":
        activation_function = ActivationFunctions.relu
    elif args.activation == "tanh":
        activation_function = ActivationFunctions.tanh
    else:
        activation_function = ActivationFunctions.identity

    nn = NeuralNetwork(input_size = 784, output_size = 10, num_hidden_layer = args.num_layers, 
                       hidden_layer_size = args.hidden_size, weight_init_startegy=args.weight_init,
                       activation_function = activation_function, 
                       output_activation = ActivationFunctions.softmax, loss_function = args.loss)
    
    model_params = nn.get_learning_params()

    if args.optimizer == "sgd":
        optimizer = SGD(args.learning_rate, args.epochs, args.batch_size, model_params, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(args.learning_rate, args.epochs, args.batch_size, model_params, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NestrovAcceleratedGradient(args.learning_rate, args.epochs, args.batch_size, model_params, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(args.learning_rate, args.epochs, args.batch_size, model_params, beta=args.beta, epsilon=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = Adam(args.learning_rate, args.epochs, args.batch_size, model_params, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == "nadam":
        optimizer = NADAM(args.learning_rate, args.epochs, args.batch_size, model_params, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid optimizer")
    
    # print("Training the model")
    nn.train(optimizer,x_train, y_train,x_val,y_val)
    # print("Training completed")

    test_prediction = nn.predict(x_test)
    confusion_matrix = wandb.plot.confusion_matrix(y_true=y_test, preds=test_prediction, class_names=class_names)
    wandb.log({"confusion_matrix": confusion_matrix})

    wandb.finish()





