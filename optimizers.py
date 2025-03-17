import numpy as np
import wandb
# from train import x_test, y_test, x_val, y_val

class SGD:
    def __init__(self, learning_rate,epochs, batch_size, model_parameters=None,weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # model = model
        self.weight_decay = weight_decay

    def update_weights(self,model ,X, y):
        model.forward(X)
        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            model.weights[i] -= self.learning_rate * weight_gradients[i] - self.weight_decay * self.learning_rate * model.weights[i]
            model.biases[i] -= self.learning_rate * bias_gradients[i]

 
class Momentum:
    def __init__(self, learning_rate, epochs, batch_size,model_parameters=None, momentum=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.velocity_biases = [np.zeros_like(b) for b in model_parameters['biases']]

    def update_weights(self, model, X, y):
        model.forward(X)
        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] +  self.learning_rate * weight_gradients[i]
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * bias_gradients[i]

            model.weights[i] -=  self.velocity_weights[i] - self.weight_decay * self.learning_rate * model.weights[i]
            model.biases[i] -=  self.velocity_biases[i] 



class NestrovAcceleratedGradient:
    def __init__(self, learning_rate, epochs, batch_size,model_parameters=None, momentum=0.9,weight_decay = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.velocity_biases = [np.zeros_like(b) for b in model_parameters['biases']]

    def update_weights(self, model, X, y):        
        weights_before_lookahead = [w.copy() for w in model.weights]
        biases_before_lookahead = [b.copy() for b in model.biases]
        model.forward(X)
        

        model.weights = [ weights - self.momentum*velocity for weights, velocity in zip(model.weights, self.velocity_weights)]
        model.biases = [ biases - self.momentum*velocity for biases, velocity in zip(model.biases, self.velocity_biases)]

        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * weight_gradients[i]
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * bias_gradients[i]

            model.weights[i] = weights_before_lookahead[i] -  self.velocity_weights[i] - self.weight_decay * self.learning_rate * model.weights[i]
            model.biases[i] = biases_before_lookahead[i] -  self.velocity_biases[i] 

class RMSProp:
    def __init__(self, learning_rate, epochs, batch_size,model_parameters=None, beta=0.9, epsilon=1e-8,weight_decay=0):
        self.inital_learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # model = model
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.accumulated_weight_gradients = [np.zeros_like(w) for w in model_parameters['weights']]
        self.accumulated_bias_gradients = [np.zeros_like(b) for b in model_parameters['biases']]

    def update_weights(self,model, X, y):
        model.forward(X)
        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            self.accumulated_weight_gradients[i] = self.beta * self.accumulated_weight_gradients[i] + (1-self.beta) * weight_gradients[i]**2
            self.accumulated_bias_gradients[i] = self.beta * self.accumulated_bias_gradients[i] + (1-self.beta) * bias_gradients[i]**2

            # print(np.sqrt(self.accumulated_weight_gradients[i]).shape)

            model.weights[i] -= self.inital_learning_rate / (np.sqrt(self.accumulated_weight_gradients[i]) + self.epsilon)*(weight_gradients[i])
            model.biases[i] -= self.inital_learning_rate * bias_gradients[i] / (np.sqrt(self.accumulated_bias_gradients[i]) + self.epsilon)

            if self.weight_decay > 0:
                model.weights[i] -= self.inital_learning_rate * self.weight_decay * model.weights[i] 



class Adam:
    def __init__(self, learning_rate, epochs, batch_size,model_parameters=None, beta1=0.9, beta2=0.999, epsilon=1e-8 , weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.weight_decay =weight_decay
        self.moment_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.moment_biases = [np.zeros_like(b) for b in model_parameters['biases']]
        self.velocity_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.velocity_biases = [np.zeros_like(b) for b in model_parameters['biases']]

    def update_weights(self, model, X, y):
        model.forward(X)
        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            self.moment_weights[i] = self.beta1 * self.moment_weights[i] + (1-self.beta1) * weight_gradients[i]
            self.moment_biases[i] = self.beta1 * self.moment_biases[i] + (1-self.beta1) * bias_gradients[i]

            self.velocity_weights[i] = self.beta2 * self.velocity_weights[i] + (1-self.beta2) * weight_gradients[i]**2
            self.velocity_biases[i] = self.beta2 * self.velocity_biases[i] + (1-self.beta2) * bias_gradients[i]**2

            moment_weights_hat = self.moment_weights[i] / (1 - self.beta1**(self.t+1))
            moment_biases_hat = self.moment_biases[i] / (1 - self.beta1**(self.t+1))

            velocity_weights_hat = self.velocity_weights[i] / (1 - self.beta2**(self.t+1))
            velocity_biases_hat = self.velocity_biases[i] / (1 - self.beta2**(self.t+1))

            model.weights[i] -= self.learning_rate / (np.sqrt(velocity_weights_hat) + self.epsilon)* (moment_weights_hat)
            model.biases[i] -= self.learning_rate * moment_biases_hat / (np.sqrt(velocity_biases_hat) + self.epsilon)
            self.t+=1

            if self.weight_decay >0:
                model.weights[i] -= self.learning_rate * self.weight_decay * model.weights[i]

    

class NADAM:
    def __init__(self, learning_rate, epochs, batch_size,model_parameters=None, beta1=0.9, beta2=0.999, epsilon=1e-8,weight_decay=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t=0
        self.moment_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.moment_biases = [np.zeros_like(b) for b in model_parameters['biases']]
        self.velocity_weights = [np.zeros_like(w) for w in model_parameters['weights']]
        self.velocity_biases = [np.zeros_like(b) for b in model_parameters['biases']]

    def update_weights(self,model, X, y):
        model.forward(X)
        weight_gradients, bias_gradients = model.backward(X,y)

        for i in range(len(model.weights)):
            self.moment_weights[i] = self.beta1 * self.moment_weights[i] + (1-self.beta1) * weight_gradients[i]
            self.moment_biases[i] = self.beta1 * self.moment_biases[i] + (1-self.beta1) * bias_gradients[i]

            self.velocity_weights[i] = self.beta2 * self.velocity_weights[i] + (1-self.beta2) * weight_gradients[i]**2
            self.velocity_biases[i] = self.beta2 * self.velocity_biases[i] + (1-self.beta2) * bias_gradients[i]**2

            moment_weights_hat = self.moment_weights[i] / (1 - self.beta1**(self.t+1))
            moment_biases_hat = self.moment_biases[i] / (1 - self.beta1**(self.t+1))

            velocity_weights_hat = self.velocity_weights[i] / (1 - self.beta2**(self.t+1))
            velocity_biases_hat = self.velocity_biases[i] / (1 - self.beta2**(self.t+1))

            model.weights[i] -= self.learning_rate / (np.sqrt(velocity_weights_hat) + self.epsilon) * ((self.beta1*moment_weights_hat + (1-self.beta1)*weight_gradients[i]/(1-self.beta1**(self.t+1))))
            model.biases[i] -= self.learning_rate * (self.beta1*moment_biases_hat + (1-self.beta1)*bias_gradients[i]/(1-self.beta1**(self.t+1))) / (np.sqrt(velocity_biases_hat) + self.epsilon)

            self.t+=1

            if self.weight_decay>0:
                model.weights[i] -=  self.weight_decay* self.learning_rate * model.weights[i]
