# DA6401 Assignment 1

#### Report Link (WANDB)
Project report for this assignment can be accessed here: [Click here to view the wandb report](https://wandb.ai/rohitrk06-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1--VmlldzoxMTUyNzQyNQ?accessToken=kwpx9fcqk9nhnc145j6qpynpwemfveonc24ffyydd4if54a1ebcrveu20ddp30gt)

#### GitHub Repository
Assignment Github repository can be accessed here: [Click here visit github repo](https://github.com/rohitrk06/da6401_assignment1)

## Assignment Folder Structure

Below files are main submission files for the assignment.
```
.
├── fashion_mnist.py                    # Loads and visualizes the Fashion MNIST dataset
├── helper_functions.py                 # Utility functions (data preprocessing, activation functions, etc.)
├── neural_network.py                   # Implementation of the neural network
├── optimizers.py                       # Implementation of different optimizers (SGD, Adam, RMSProp, etc.)
├── train.py                            # Main script for training the neural network
├── config.yaml                         # Configuration file for running wandb sweeps
├── second_sweep_config.yaml            # Configuration file for sweeps using Bayesian Strategy for wandb sweeps top hyperparameter of initial sweep
└── README.md              # Documentation

```

Additional files are as follows:
```
.
├── timestampLogger.py     # Utility for logging timestamps
├── requirements.txt        # Requirements

```


## Running the code

Training script can be run using the following command with the desired hyperparameters as arguments. For example, to run the training script with the following hyperparameters:
- Activation function: tanh
- Batch size: 128
- Number of epochs: 5
- Hidden size: 64
- Learning rate: 0.0006
- Loss function: mean_squared_error
- Number of layers: 5
- Optimizer: rmsprop
- Weight decay: 0.0005
- Weight initialization: xavier

```
python train.py --activation=tanh --batch_size=128 --epochs=5 --hidden_size=64 --learning_rate=0.0006 --loss=mean_squared_error --num_layers=5 --optimizer=rmsprop --weight_decay=0.0005 --weight_init=xavier
```
#### Command-Line Arguments

| Argument              | Description |
|-----------------------|-------------|
| `--dataset`          | Choose dataset: `fashion_mnist` or `mnist` |
| `--epochs`           | Number of training epochs |
| `--batch_size`       | Mini-batch size |
| `--loss`             | Loss function: `cross_entropy` or `mean_squared_error` |
| `--optimizer`        | Optimizer: `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `--learning_rate`    | Learning rate for the optimizer |
| `--momentum`         | Momentum factor (for applicable optimizers) |
| `--beta1` & `--beta2` | Beta values for Adam/Nadam optimizers |
| `--weight_decay`     | Regularization term |
| `--num_layers`       | Number of hidden layers |
| `--hidden_size`      | Number of neurons in hidden layers |
| `--activation`       | Activation function: `sigmoid`, `relu`, `tanh`, `identity` |
| `--weight_init`      | Weight initialization method: `random`, `xavier` |
| `--wandb_project`    | WANDB project name |
| `--wandb_entity`     | WANDB entity for experiment tracking |

Before running the above command, make sure to install the required packages using the following command:

```
pip install -r requirements.txt
```

## Wandb Sweeps

We have used wandb sweeps to tune the hyperparameters of the neural network. The configuration file for the sweeps can be found in `config.yaml`. The sweeps can be run using the following command:

``` bash
wandb sweep config.yaml
wandb agent rohitrk06-indian-institute-of-technology-madras/da6401_assignment1/h5mzihh7 
```

## Code Structure

- `fashion_mnist.py`: Loads the Fashion MNIST dataset, visualizes images, and logs data to WandB.
- `helper_functions.py`: Contains utility functions for data preprocessing, normalization, one-hot encoding, and activation functions.
- `neural_network.py`: Implements a fully connected neural network with forward and backward propagation.
    - Supports configurable activation functions.
    - Includes methods for weight initialization using random and Xavier initialization.
    - Computes loss and accuracy.
- `optimizers.py`: Implements multiple optimizers, including:
    - Stochastic Gradient Descent (SGD)
    - Momentum-based optimization
    - Nesterov Accelerated Gradient (NAG)
    - RMSProp, Adam, and Nadam optimizers
    - Any new optimizer can be added by writing a new class that has `update_weights` method.
- `train.py`: Main script for training the neural network.
    - Parses command-line arguments for training configuration.
    - Loads and preprocesses the dataset.
    - Initializes the neural network and optimizer.
    - Trains the model and logs results to WandB.
    - Evaluates the model and logs test results


## Submitted by
Rohit Kumar (DA24S003)



