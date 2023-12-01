# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pandas as pd
from preprocessing import tokenizer, document_term_matrix, get_dictionary, dataset, glove_embeddings, word2vec_embeddings
from evaluation.metrics import CoherenceScores
from sklearn.preprocessing import normalize
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HeiOnline.csv', help='the path to the dataset')
parser.add_argument('--vectorizer', type=str, default='tfidf', help='the TfidfVectorizer from sklearn')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--num_topics', type=int, default=20, help='the number of topics')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
parser.add_argument('--epochs', type=int, default=100, help='the number of the training iterations')
parser.add_argument('--batch_size', type=int, default=64, help='the size of the batches')
parser.add_argument('--lr', type=float, default=0.002, help='the learning rate of Adam')
parser.add_argument('--b1', type=float, default=0.9, help='the decay of first order momentum of gradient for Adam')
parser.add_argument('--b2', type=float, default=0.999, help='the decay of second order momentum of gradient for Adam')
parser.add_argument('--embeddings', type=str, default='Word2Vec', help='Word2Vec or GloVe')
parser.add_argument('--decay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--hidden_size', type=int, default=800, help="the hidden layer's size of ETM")
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: 1 for skip-gram, 0 for CBOW.')
opt = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ETM(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size, embeddings, weights, device):
        super(ETM, self).__init__()
        self.batch_size = batch_size
        self.device = device

        if embeddings == 'Word2Vec':
            self.rho = nn.Embedding.from_pretrained(weights)
            self.alphas = nn.Linear(100, num_topics, bias=False)
        else:
            self.rho = nn.Embedding.from_pretrained(weights)
            self.alphas = nn.Linear(300, num_topics, bias=False)

        self.encoder = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.Softplus(),
        nn.Linear(hidden_size, hidden_size),
        nn.Softplus()
        )

        self.mean = nn.Linear(hidden_size, num_topics)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.logvar = nn.Linear(hidden_size, num_topics)
        self.logvar_bn = nn.BatchNorm1d(num_topics)

        self.decoder_bn = nn.BatchNorm1d(vocab_size)

    def encode(self, inputs):
        x = self.encoder(inputs)

        posterior_mean = self.mean_bn(self.mean(x))
        posterior_logvar = self.logvar_bn(self.logvar(x))

        KL_divergence = 0.5*torch.sum(1 + posterior_logvar - posterior_mean**2 - torch.exp(posterior_logvar), 1)

        return posterior_mean, posterior_logvar, torch.mean(KL_divergence)

    def reparameterization(self, posterior_mean, posterior_logvar):
        epsilon = torch.randn_like(posterior_logvar, device=self.device)
        z = posterior_mean + torch.sqrt(torch.exp(posterior_logvar))*epsilon

        return z

    def get_beta(self):
        beta = F.softmax(self.alphas(self.rho.weight), 0)
        return beta.transpose(1, 0)

    def get_theta(self, normalized_inputs):
        mean, logvar, KL_divergence = self.encode(normalized_inputs)
        z = self.reparameterization(mean, logvar)
        theta = F.softmax(z, 1)

        return theta, KL_divergence

    def decode(self, theta, beta):
        result = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), 1)
        prediction = torch.log(result)

        return prediction

    def forward(self, inputs, normalized_inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size

        beta = self.get_beta()
        theta, KL_divergence = self.get_theta(normalized_inputs)
        output = self.decode(theta, beta)

        reconstruction_loss = - torch.sum(output*inputs, 1)

        return output, torch.mean(reconstruction_loss) + KL_divergence


def train_model(train_loader, model, optimizer, epochs, device):

    """Return a list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

        Arguments:

            train_loader: An iterable over the dataset.
            model: The ETM model.
            optimizer: The optimizer for updating ETM's paratemeters.
            epochs: The number of the training iterations.
            device: 'cuda' or 'cpu'.

        Returns:

            train_losses: A list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

    """

    model.train()

    for epoch in range(epochs):
        losses = []
        train_losses = []
        total = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            normalized_inputs = (inputs/(torch.sum(inputs, 1).unsqueeze(1))).to(device)

            model.zero_grad()
            output, loss = model(inputs, normalized_inputs)
            loss.backward()
            optimizer.step()

            losses.append(loss)
            total += 1

        epoch_loss = sum(losses)/total
        train_losses.append(epoch_loss.item())
        print(f'Epoch {epoch + 1}/{epochs}, Loss={epoch_loss}')

    return train_losses


def get_topics(model, tfidf, num_topics, top_words):

    """Returns a list of lists of the top words for each topic.

        Arguments:

            model: The ETM model.
            tfidf: The TfidfVectorizer from preprocessing.py.
            num_topics: The number of topics.
            top_words: The number of the top words for each topics.

        Returns:

            topic_list: A list of lists containing the top words for each topic.

    """

# Generate the topic-word matrix
    beta = model.get_beta()
# Create a list of lists of the top words for each topic  
    topic_list = []

    for topic in beta:
        topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    df.to_excel(f'ETM_{num_topics}.xlsx')

    return topic_list


def get_doc_topic_list(model, train_loader, device):

    """Return the list of the topic of each document.

        Arguments:

            model: The ETM model.
            train_loader: An iterable over the dataset.
            device: 'cpu' or 'cuda'

        Returns:

            doc_topic_list: A list of the topic assigned to each document by ETM.        

    """

    model.eval()
    flag = True
    
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            normalized_inputs = (inputs/(torch.sum(inputs, 1).unsqueeze(1))).to(device)

            doc_topic_dist, _ = model.get_theta(normalized_inputs)

            if flag == True:
                doc_topic_matrix = doc_topic_dist.argmax(axis=1)
                flag = False
            else:
                doc_topic_matrix = torch.cat((doc_topic_matrix, doc_topic_dist.argmax(axis=1)), axis=0)

    doc_topic_list = np.array(doc_topic_matrix.cpu())   
    return doc_topic_list


if __name__ == '__main__':
# Define the dataset and the arguments
	df = pd.read_csv(opt.dataset)
	articles = df['content']

# Generate the document term matrix and the vectorizer
    processed_articles = articles.apply(tokenizer)
    tfidf, dtm = document_term_matrix(processed_articles, opt.vectorizer, opt.min_df, opt.max_df)
    dtm = normalize(dtm)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
    bow, dictionary, w2v = get_dictionary(cv, articles, opt.min_df, opt.size, opt.sg)

    if opt.embeddings == 'GloVe':
# Load the GloVe embeddings
        embeddings_dict = {}

        with open("glove.42B.300d.txt", 'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

# Create a matrix containing the GloVe embeddings
        embedding_matrix = glove_embeddings(tfidf, vocab_size, embeddings_dict)
    else:
# Create a matrix containing the Word2Vec embeddings
        embedding_matrix = word2vec_embeddings(tfidf, vocab_size, w2v)

# Make the embedding matrix a float tensor to be used as rho's weights
    weights = torch.FloatTensor(embedding_matrix)

# Create the train loader
    train_loader = dataset(dtm, opt.batch_size)

# Define the models and the optimizers
    vocab_size = dtm.shape[1]
    model = (ETM(vocab_size, opt.hidden_size, opt.num_topics, opt.batch_size, opt.embeddings, weights, device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.decay)

# Train the model
    train_model(train_loader, model, optimizer, opt.epochs, device)

# Create the list of lists of the top 10 words of each topic
    topic_list = get_topics(model, tfidf, opt.num_topics, opt.top_words)

# Print the title of the document and its topic based on ETM
    doc_topic_list = get_doc_topic_list(model, train_loader, device)
    df['Topic'] = doc_topic_list
    print(df[['title', 'Topic']])

# Calculate the coherence scores
    evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
    coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
    print(coherence_scores)