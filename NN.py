import numpy as np
import pdb
import random

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

    def SGD(self, training_data, n_epochs, mini_batch_size, eta, test_data=None):
        #
        n = len(training_data)
        
        # for each epoch that we want to train for
        for nth_epoch in range(n_epochs):
            # shuffle the training data and create mini batches
            random.shuffle(training_data)
            # create mini batches by indexing into the training data
            # the indexes are from 0 to n with mini_batch_size steps
            # and each mini_batch_size'd chunk is one mini batch
            mini_batches = [
                    training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
            # now, for each mini batch, do the update step using the update_mini_batch function
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # if test data is provided
            if test_data:
                n_test = len(test_data)
                # print the training status
                print("Epoch ", nth_epoch, ": ", self.evaluate(test_data), "/", n_test)
            else:
                print("Epoch ", nth_epoch, " completed.")

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
            # new activation is the sigmoid of the calculated input
            activation = sigmoid(z)
            activations.append(activation)  # add a vector to the list

        # backward pass
        

        # return the partial derivatives
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        # return the number of digits that were correctly classified

        # the output of the neural network is the max value of activation in the final layer
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # count the number of times in test_results that the computed value is equal to the
        # label
        return sum(int(x == y) for (x, y) in test_results)

# misc helper functions
def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):       # derivative of the sigmoid function
    return sigmoid(z)*(1 - sigmoid(z))
def cost_derivative(self, output_activations, y):
    return (output_activations - y) # return the vector of partial(C_x) / partial(a) for the output
