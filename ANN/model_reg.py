import numpy as np
import pandas as pd
import random


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1, 0)


class NNetwork(object):
    def __init__(self, sizes, optimizer="sgd", activation="sigmoid", xavier=True):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.optimizer = optimizer
        self.activation_function = sigmoid if activation == "sigmoid" else relu
        self.activation_prime = sigmoid_prime if activation == "sigmoid" else relu_prime
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if xavier:
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        if optimizer == "ADAM":
            # ADAM specific parameters
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            # time step
            self.t = 0

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta_learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if self.optimizer == "SGD":
            self.weights = [w - (eta_learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta_learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        elif self.optimizer == "ADAM":
            self.t += 1
            self.m_w = [self.beta1 * m + (1-self.beta1) * nw for m, nw in zip(self.m_w, nabla_w)]
            self.v_w = [self.beta2 * v + (1-self.beta2) * (nw**2) for v, nw in zip(self.v_w, nabla_w)]
            self.m_b = [self.beta1 * m + (1-self.beta1) * nb for m, nb in zip(self.m_b, nabla_b)]
            self.v_b = [self.beta2 * v + (1-self.beta2) * (nb**2) for v, nb in zip(self.v_b, nabla_b)]

            m_w_hat = [m/(1-self.beta1**self.t) for m in self.m_w]
            v_w_hat = [v/(1-self.beta2**self.t) for v in self.v_w]
            m_b_hat = [m/(1-self.beta1**self.t) for m in self.m_b]
            v_b_hat = [v/(1-self.beta2**self.t) for v in self.v_b]

            self.weights = [w - (eta_learning_rate*mw)/(np.sqrt(vw) + self.epsilon) for w, mw, vw in
                            zip(self.weights, m_w_hat, v_w_hat)]
            self.biases = [b - (eta_learning_rate*mb)/(np.sqrt(vb) + self.epsilon) for b, mb, vb in
                           zip(self.biases, m_b_hat, v_b_hat)]


    def sgd(self, training_data, epochs, mini_batch_size, eta_learning_rate, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta_learning_rate)
            if test_data and j % 10 == 0:
                print(f'Epoch {j}: MSE on test data: {self.evaluate(test_data)}')

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        return np.mean([((x - y) ** 2) for (x, y) in test_results])


def load_data():
    df = pd.read_csv(filepath_or_buffer='data/auto-mpg.csv', na_values=["NA","?"])
    df = df.dropna()
    x = df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin"]].values.astype(np.float32)
    y = df["mpg"].values.astype(np.float32)
    y = y.reshape(-1, 1)
    return x, y


def prepare_data(x, y):
    data = [(x[i].reshape(-1, 1), y[i].reshape(-1, 1)) for i in range(x.shape[0])]
    random.shuffle(data)
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
    return train_data, test_data


x, y = load_data()
train_data, test_data = prepare_data(x, y)

net = NNetwork([7, 50, 25, 1], optimizer="SGD", activation="sigmoid")
net.sgd(train_data, 1000, 30, 0.0001, test_data=test_data)