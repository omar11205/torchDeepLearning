import numpy as np
import random
import pickle
import gzip

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class NNetwork(object):
    """
    (*) operator for numpy arrays or matrices performs element-wise multiplication
    """

    def __init__(self, sizes):
        """
        sizes: a LIST containing the number of neurons in the respective layers np.random.randn generates
        gaussian distributions with mean 0 and STD of 1 list[1:] slices elements from index 1 to the end of the list,
        (exclude the first layer of neurons to have bias, the input layer don't have bias) zip() combines multiple
        iterables in a tuple list[:-1] slices all the elements in the list least the last one (the output layer don't
        have weights connected to a next layer)
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # with (y, 1) generates a list of numpy column vectors
        self.weights = [np.random.randn(y, x) for x, y in
                        zip(sizes[:-1], sizes[1:])]  # generate a list of numpy matrices

    def feedforward(self, a):
        """Return the output of the network if 'a' is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result. Note that the
        neural network's output is assumed to be the index of whichever neuron in the final layer has the highest
        activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives 'partial C_x / partial a' for the output activations"""
        return (output_activations - y)

    def backprop(self, x, y):
        """Return a tuple '(nabla_c, nabla_w)' representing the gradient for the cost function C_x. 'nabla_b' and
        'nabla_w' are layer-by-layer lists of numpy arrays, similar to 'self.biases' and 'self.weights'. x: contains
        the input layer column vector, y: contains the associated desired result column vector"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Step 1. Input x: Set the corresponding activation a^1 for the input layer.
        activation = x  # the activation vector of the input layer is the input layer vector itself
        activations = [x]  # list to store all the activations, layer by layer

        # Step 2. Feedforward, (forward pass): For each l = 2, 3, ..., L
        # compute z^{l} = (w^l DOT PRODUCT a^{l-1})+ b^l and a^{l} = sigma(z^{l})
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Step 3. Output error delta^L: Compute the vector delta^L = nabla_a_C * sigma'(z^L)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # Step 4. Backpropagate the error: For each l = L-1, L-2, ..., L-N compute
        # delta^{l} = ((w^{l+1})^T DOT PRODUCT delta^{l+1}) * sigma'(z^{l})
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """Note that the variable 1 in the loop below is used a little differently to the notation in Chapter 2 of 
        the book. Here, l = 1 means the last layer of neurons, l = 2 is the second-last layer of neurons, and so on. 
        It's a renumbering of the scheme in the book, used here to take advantage of the fact that python can use 
        negative indices in lists."""
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta_learning_rate):
        """ Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
         The 'mini_batch' is a list of tuples '(x,y)' """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta_learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta_learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def sgd(self, training_data, epochs, mini_batch_size, eta_learning_rate, test_data=None):
        """ Train the neural network using mini-batch Stochastic Gradient Descent. The 'training_data' is a list of
        tuples (x,y) representing the training inputs and the desired outputs. The other non-optional parameters are
        self-explanatory. If 'test_data' is provided, the network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is usefully for taking progress, but slows down the training"""

        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta_learning_rate)
            print("bias sample b[0][5]", self.biases[0][5])
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')


def mnist_import_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def mnist_one_hot_encoding(j):
    """Transform an integer result to the desired training result, that is a 10-dimensional column vector"""
    e = np.zeros((10,1))
    e[j] = 1.0
    return e


def mnist_data_wrapper(show_lengths=False):
    tr_d, va_d, te_d = mnist_import_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [mnist_one_hot_encoding(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    if show_lengths:
        print(f'Training data length: {len(list(training_inputs))}')
        print(f'Validation data length: {len(validation_inputs)}')
        print(f'Test data length: {len(test_inputs)}')
    return (training_data, validation_data, test_data)


# training_data, validation_data, test_data = mnist_data_wrapper(show_lengths=True)

# net = NNetwork([784, 30, 10])
# net.sgd(training_data, 30, 10, 0.5, test_data=test_data)



