"""
Simple PyTorch classifier for MNIST, demonstrating 
BAD comment practice.
"""

# importing packages
import torch
import torch.nn as nn
import numpy as np

# setting the random seed
torch.manual_seed(0)

### IMPORTING DATA ###

# skip first 16 bytes for image file data
X_train = np.fromfile('mnist/mnist-train-images.dat', dtype=np.uint8, offset=16)

# skip the first 8 bytes
y_train = np.fromfile('mnist/mnist-train-labels.dat', dtype=np.uint8, offset=8)

# skip first 16 bytes for image file data
X_test = np.fromfile('mnist/mnist-test-images.dat', dtype=np.uint8, offset=16)

# skip first 16 bytes for image file data
y_test = np.fromfile('mnist/mnist-test-labels.dat', dtype=np.uint8, offset=8)

# split the training data into individual images
X_train = X_train.reshape(-1, 784)

X_test = X_test.reshape(-1, 784)


### BUILDING MODEL ###

# construct the neural network
model = nn.Sequential(
                nn.Linear(784, 256), # 256 hidden units
                nn.ReLU(),
                nn.Linear(256, 10) # predicting 10 classes
                # nn.Softmax()
                )

# Adam is an optimization algorithm that can be used instead of the classical
# stochastic gradient descent procedure to update network weights iterative
# based in training data.  Adam was presented by Diederik Kingma from OpenAI
# and Jimmy Ba from the University of Toronto in their 2015 ICLR paper (poster)
# titled "Adam: A Method for Stochastic Optimization" (From
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

### RUNNING TRAINING ### 
# number of epochs
N = 1000


# loops for 1000 epochs
for i in range(N):
    # sampling a minibatch of size 32 from the training data
    indices = torch.randperm(len(X_train))[:32]
    X_mb, y_mb = X_train[indices], y_train[indices]

    # compute predictions as logits for CE loss
    logit_preds = model(torch.tensor(X_mb, dtype=torch.float))

    # ce needs integer as target 
    loss = nn.functional.cross_entropy(logit_preds, torch.tensor(y_mb, dtype=torch.long))

    # zero the grad
    optim.zero_grad()
    # call backward on the loss
    loss.backward()
    # take a step with optim
    optim.step()

# compute predictions for the test images
preds = model(torch.tensor(X_test, dtype=torch.float))

# get actual predictions
preds = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1)

# Compute the accuracy between the predictions and the targets.
def compute_accuracy(predictions, targets):
    """
    :param predictions (iterable): The predictions.
    :param targets (iterable): The targets.
    :return (float): The accuracy of the predictions.
    """

accuracy = sum([pred == actual for pred, actual in zip(preds, y_test)]) / len(preds)
print(accuracy)
