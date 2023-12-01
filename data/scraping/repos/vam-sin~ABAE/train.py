import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import nltk
nltk.download('punkt')
from src.sent_embeddings import calculate_ys, calculate_zs
from src.sent_reconstruction import calculate_rs
from src.hinge_loss import regularized_loss_value
from src.aspect_retrieval import retrieve_aspects
from src.coherence_score import coherence_score
from torch.autograd import Variable
import random

# Tasks
# 1. Use the functions to get the aspects. Done
# 2. Reconstruct the sentence. Done
# 3. Convert to tensors. Done
# 4. Loss function optimization. Done
# 5. Implement the negative sampling part in training.

# Hyperparameters
epochs = 100
l = 1 # lambda hyperparameter
lr = 0.001 # learning rate
m = 30 # negative samples for each input

# Model
class ABAE_Model(nn.Module):

    def __init__(self, M, E, unique_words, T, W, b):
        super(ABAE_Model, self).__init__()
        self.M = M
        self.E = E
        self.unique_words = unique_words
        self.T = T
        self.W = W
        self.b = b

    def forward(self, input):
        # Forward pass of the review given
        ys = calculate_ys(input, self.E, self.unique_words)
        zs = calculate_zs(input, self.M, ys, self.E, self.unique_words)
        rs = calculate_rs(self.T, self.W, zs, self.b)

        return rs, ys, zs


# Parameters
M = Variable(torch.randn(200, 200).type(torch.float32), requires_grad=True)
W = Variable(torch.randn(200, 200).type(torch.float32), requires_grad=True)
b = Variable(torch.randn(200, 1).type(torch.float32), requires_grad=True)

infile = open('src/E.pickle','rb')
E = pickle.load(infile)
infile.close()

infile = open('src/reviews.pickle','rb')
reviews = pickle.load(infile)
infile.close()

infile = open('src/unique_words.pickle','rb')
unique_words = pickle.load(infile)
infile.close()

infile = open('src/T.pickle','rb')
T = pickle.load(infile)
infile.close()
T = np.asarray(T)
T = torch.tensor(T, requires_grad=True)

# Model Instantiation
abae = ABAE_Model(M, E, unique_words, T, W, b)

optimizer = optim.Adam([M, W, b, T], lr=lr)

# Training of the model
for i in range(epochs):
    random.shuffle(reviews)
    epoch_loss = 0.0
    optimizer.zero_grad()
    print("Epoch Number: " + str(i))
    j = 0
    while (len(reviews)-m)-j >= 0:
        # Negative Sampling
        loss = 0.0
        list_rs = []
        list_ys = []
        list_zs = []
        for k in range(j, j+m):
            rs, ys, zs = abae(reviews[k])
            list_rs.append(rs)
            list_ys.append(ys)
            list_zs.append(zs)

        rev_loss = regularized_loss_value(list_rs, list_zs, list_ys, T, l)
        loss += rev_loss.sum()
        epoch_loss += rev_loss.sum()

        loss.sum().backward()
        optimizer.step()
        # print(j)
        j += m

    print("Epoch Loss: " + str(epoch_loss) +"\n")

# The T matrix represents the 14 aspects predicted
outfile = open('aspects_emeddings.pickle','wb')
pickle.dump(T ,outfile)
outfile.close()

# Print all the aspects
retrieve_aspects(T, 1)

# Calculate Coherence Score
T = T.detach().numpy()
cs = coherence_score(T, 5, reviews)
