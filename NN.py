import numpy as np

class Network:
    # Class definition of a Network object which represents a neural network

    def __init__(self, sizes, verbose=False):
        # Constructor for a Network object
        # This will simply create the weights and biases arrays for the specified sizes
        # The weights and biases are randomly initialized

        self.num_layers = len(sizes)
        self.sizes = sizes

        # We only need biases for layers AFTER the first layer, therefore we look at the
        # sizes from index 1 onwards. 
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x, y in zip(sizes[1:], sizes[:-1])]

        if verbose:
            print("Number of layers: ", self.num_layers)
            print("Layer sizes: ", self.sizes)
            print("Size of matrix of weights: ", [w.shape for w in self.weights]) 
            print("Size of matrix of biases: ", [b.shape for b in self.biases]) 

    def feedforward(self, a):
        # computes a forward pass through the network give input "a"
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
