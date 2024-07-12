import torch
# import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn

# creating a column matrix of 50 linearly spaced elements
X = torch.linspace(1, 50, 50).reshape(-1, 1)

# creating a random array of error values of the same shape of the X column matrix (50,1)
torch.manual_seed(71)
e = torch.randint(-8, 9, (50,1), dtype=torch.float)

# creating a tensor y(x) = 2*x + 1 plus an random error e (noise)
y = 2*X + 1 + e

# in order to plot, first convert the tensors into numpy arrays
# plt.scatter(X.numpy(), y.numpy())
# plt.show()

# creating a program for linear regression with NN
torch.manual_seed(59)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()  # inherits the constructor method of the superior object (nn.Module)
        # the type of neural network layer that is employed here: Linear is a fully connected or Dense layer
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# instantiating a model object, one simple in feature x, one simple out feature y
model = Model(1,1)

# setting the loss function: mean square error loss (the target function for the optimization)
criterion = nn.MSELoss()
print(criterion.type)

# setting the learning rate as stochastic gradient descend, learning rate, lr = 0.01
# this could be used to update the weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training algorithm

# epochs: entire pass or entire processing of the dataset that
# ends updating the biases and weights of the network.
# With this update, the next epoch will be closer and closer
# to the optimization. In order to reach an optimum number of
# epochs we need to track the losses

epochs = 25
losses = []
for epoch in range(epochs):
    epoch += 1

    # prediction of y (forward pass)
    y_pred = model.forward(X)

    # calculate the loss (error)
    loss = criterion(y_pred, y)

    # record the error
    losses.append(loss.detach().numpy())

    print(f'epoch: {epoch}, loss: {loss.item()}, weight: {model.linear.weight.item()}, bias {model.linear.bias.item()}')

    # to prevent the accumulation of the gradients steps in the next epoch
    # starting "fresh" for the next epoch
    optimizer.zero_grad()

    # find the derivatives with respect to the loss function (backward step)
    loss.backward()

    # updating the weights and biases
    optimizer.step()
