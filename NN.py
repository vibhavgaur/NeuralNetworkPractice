import numpy as np
import pdb

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

    def update_mini_batch(self, mini_batch, eta):
        # update the network's weights and biases by calculating the gradient using
        # backpropagation. This is done for a mini batch at a time. eta is the learning rate
        
        # initialize arrays for the partial derivatives
        # nabla_b --> partials w.r.t. biases
        # nabla_w --> partials w.r.t. weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # for each input and label in the minibatch
        for x, y in mini_batch:
            # backpropagate through the network to calculate the derivatives
            # delta_nabla_b and delta_nabla_w are the changes to be made to
            # the weights and biases of the network from the "learning" that
            # happens due to this mini_batch
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)
            breakpoint()
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # updating the weights and biases
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        #initialize lists for the partial derivatives
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # forward pass a training example x through the network
        activation = x
        activations = [x]   # store all the activations layer by layer (in a list)
        zs = []             # store all the z vectors layer by layer

        # calculating inputs and outputs for each layer, layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass

        # return the partial derivatives
        return nabla_w, nabla_b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):       # derivative of the sigmoid function
    return sigmoid(z)*(1 - sigmoid(z))
def cost_derivative(self, output_activations, y):
    return (output_activations - y) # return the vector of partial(C_x) / partial(a) for the output
