# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from preprocessing import tokenizer, document_term_matrix, get_dictionary, dataset
from evaluation.metrics import CoherenceScores
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HeiOnline.csv', help='the path to the dataset')
parser.add_argument('--vectorizer', type=str, default='tfidf', help='the TfidfVectorizer from sklearn')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--num_topics', type=int, default=20, help='the number of topics')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
parser.add_argument('--epochs', type=int, default=200, help='the number of the training iterations')
parser.add_argument('--batch_size', type=int, default=64, help='the size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate of Adam')
parser.add_argument('--b1', type=float, default=0.5, help='the decay of first order momentum of gradient for Adam')
parser.add_argument('--b2', type=float, default=0.999, help='the decay of second order momentum of gradient for Adam')
parser.add_argument('--n_critic', type=int, default=5, help='the number of discriminator iterations per generator iteration')
parser.add_argument('--hidden_size', type=int, default=100, help="the representation layer's size")
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: 1 for skip-gram, 0 for CBOW.')
opt = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.BatchNorm1d(hidden_size, affine=False), # No trainable parameters
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_size, num_topics),
        nn.Softmax(1)
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size 

        x = self.model(inputs)
        return x


class Generator(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(num_topics, hidden_size),
        nn.BatchNorm1d(hidden_size, affine=False), # No trainable parameters
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_size, vocab_size),
        nn.Softmax(1)
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size 

        x = self.model(inputs)
        return x


class Discriminator(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_topics, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
        nn.Linear(vocab_size + num_topics, hidden_size),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.batch_size != batch_size:
            self.batch_size = batch_size

        x = self.model(inputs)
        return x


def train_model(discriminator, generator, encoder, optimizer_d, optimizer_g, optimizer_e, epochs, num_topics, n_critic, device):

    """Return a list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

        Arguments:

            discriminator: The Discriminator.
            generator: The Generator.
            encoder: The Encoder.
            optimizer_d: The optimizer for updating the Discriminator's paratemeters.
            optimizer_g: The optimizer for updating the Generator's paratemeters.
            optimizer_e: The optimizer for updating the Encoder's paratemeters.
            epochs: The number of the training iterations.
            num_topics: The number of topics.
            n_critic: The number of discriminator iterations per generator iteration
            device: 'cpu' or 'cuda'.

        Returns:

            train_losses: A list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

    """

    discriminator.train()
    generator.train()
    encoder.train()

    for epoch in range(epochs):
        losses_d, losses_g, losses_e = [], [], []
        total_losses = []
        total, total_i = 0, 0
           
        for i, (d_r, _) in enumerate(train_loader):
            d_r = (d_r/(torch.sum(d_r, 1).unsqueeze(1))).to(device) # Normalize the inputs
            dirichlet = torch.distributions.Dirichlet(torch.ones(size=(d_r.shape[0], num_topics)))
            theta_f = (dirichlet.sample()).to(device)

            d_f, theta_r = generator(theta_f).detach(), encoder(d_r).detach()
            p_r, p_f = (torch.cat((theta_r, d_r), 1)).to(device), (torch.cat((theta_f, d_f), 1)).to(device)

            discriminator.zero_grad()
            L_d = torch.mean(discriminator(p_f)) - torch.mean(discriminator(p_r))
            L_d.backward()
            optimizer_d.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01) # Clip the weights of the discriminator

            if i%n_critic == 0:
                generator.zero_grad()
                L_g = - torch.mean(discriminator(p_f))
                L_g.backward()
                optimizer_g.step()
                losses_g.append(L_g)

                encoder.zero_grad()
                L_e = torch.mean(discriminator(p_r))
                L_e.backward()
                optimizer_e.step()
                losses_e.append(L_e)

                losses_d.append(L_d)
                total += 1

        epoch_d = sum(losses_d)/total
        epoch_g = sum(losses_g)/total
        epoch_e = sum(losses_e)/total
        total_losses.append([epoch_d.item(), epoch_g.item(), epoch_e.item()])
        print(f'Epoch {epoch + 1}/{epochs}, Encoder Loss:{epoch_e}, Generator Loss:{epoch_g}, Discriminator Loss:{epoch_d}')

    return train_losses


def get_topics(tfidf, model, num_topics, top_words, device):

    """Returns a list of lists of the top words for each topic.

        Arguments:

            tfidf: The TfidfVectorizer from preprocessing.py.
            model: The Generator.
            num_topics: The number of topics.
            top_words: The number of the top words for each topics.
            device: 'cpu' or 'cuda'.

        Returns:

            topic_list: A list of lists containing the top words for each topic.

    """

    model.eval()

    with torch.no_grad:
# Generate the topic-word matrix
        onehot_topic = torch.eye(num_topics, device=device)
        topic_word_matrix = model(onehot_topic)
# Create a list of lists of the top words for each topic
    topic_list = []

    for topic in topic_word_matrix:
        topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    df.to_excel(f'BAT_{num_topics}.xlsx')

    return topic_list

def get_doc_topic_list(dtm, model):

    """Return of list of the topic of each document.

        Arguments:

            dtm: An array representing the document term matrix.
            model: The Encoder.

        Returns:

            doc_topic_list: A list of the topics assigned to each document by the Encoder.

    """
    
    model.eval()

    with torch.no_grad():
        doc_topic_matrix = model(dtm)
        doc_topic_list = doc_topic_matrix.argmax(axis=1)

    return doc_topic_list


if __name__ == '__main__':
# Define the dataset and the arguments
	df = pd.read_csv(opt.dataset)
	articles = df['content']

# Generate the document term matrix and the vectorizer
    processed_articles = articles.apply(tokenizer)
    tfidf, dtm = document_term_matrix(processed_articles, opt.vectorizer, opt.min_df, opt.max_df)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
    bow, dictionary, w2v = get_dictionary(cv, articles, opt.min_df, opt.size, opt.sg)

# Create the train loader
    train_loader = dataset(dtm, batch_size)

# Define the models and the optimizers
    vocab_size = dtm.shape[1]
    encoder = Encoder(vocab_size, opt.hidden_size, opt.num_topics, opt.batch_size).to(device)
    generator = Generator(vocab_size, opt.hidden_size , opt.num_topics, opt.batch_size).to(device)
    discriminator = Discriminator(vocab_size, opt.hidden_size , opt.num_topics, opt.batch_size).to(device)

    optimizer_e = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Train the model
    train_model(discriminator, generator, encoder, optimizer_d, optimizer_g, optimizer_e, opt.epochs, opt.num_topics, opt.n_critic, device)

# Create the list of lists of the top 10 words of each topic
    topic_list = get_topics(tfidf, generator, opt.num_topics, opt.top_words, device)

# Print the title of the document and its topic based on BAT
    doc_topic_list = get_doc_topic_list(dtm, encoder)
    df['Topic'] = doc_topic_list
    print(df[['title', 'Topic']])

# Calculate the coherence scores
    evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
    coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
    print(coherence_scores)