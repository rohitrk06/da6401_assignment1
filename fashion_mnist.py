from keras.datasets import fashion_mnist
import wandb
import random
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if __name__ == "__main__":
    run = wandb.init(
        project = "da6401_assignment1",
        notes = "Loading and Visualising Fashion MNIST dataset",
        job_type="data-visulaization"
    )
    
    class_counts = np.bincount(y_train)

    #Class names taken from the dataset documentation
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 
    
    #Visualizing the images from each classes and logging it to wandb 
    for _ in range(5):
        idx = np.random.randint(0, min(class_counts))
        run.log({"examples": [wandb.Image(x_train[y_train == i][idx], caption=class_names[i]) for i in range(10)]})


