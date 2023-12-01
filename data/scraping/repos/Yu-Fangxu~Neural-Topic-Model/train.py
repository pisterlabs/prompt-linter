from model import NVDM, ProdLDA
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import joblib
from gensim.corpora import Dictionary
from tqdm import tqdm
import torch
import torch.nn as nn
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
dataset = "ohsumed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
n_topic = 20

def show_topic_words(word_dist, dictionary, topic_id=None, topK=20, ):
    topic_words = []
    word_dist = torch.softmax(word_dist, dim=1)
    vals, indices = torch.topk(word_dist, topK, dim=1)
    vals = vals.cpu().tolist()
    indices = indices.cpu().tolist()
    # print(indices)
    # if id2token == None and dictionary != None:
    #     id2token = {v: k for k, v in dictionary.token2id.items()}
    id2token = dictionary
    if topic_id == None:
        for i in range(n_topic):
            topic_words.append([id2token[idx] for idx in indices[i]])
    else:
        topic_words.append([id2token[idx] for idx in indices[topic_id]])
    return topic_words

def evaluate(logits, test_data, dictionary):
    topic_words = show_topic_words(logits, dictionary)
    # print(topic_words)
    return calc_topic_diversity(topic_words)

def calc_topic_coherence(topic_words, test_data, dictionary):
    cv_coherence_model = CoherenceModel(topics=topic_words, dictionary=dictionary, texts=test_data, coherence='c_v')
    # cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    return cv_score
def calc_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div
class BoWDataset(Dataset):
    def __init__(self, dataset, labels):
        self.BoW = dataset
        self.labels = labels

        self.word_count = []
        for i in range(len(self.BoW)):
            self.word_count.append(torch.sum(self.BoW[i]))
                
    def __getitem__(self, index):
        return self.BoW[index], self.labels[index], self.word_count[index]

    def __len__(self):
        return len(self.BoW)

# with open(f"temp/{dataset}.texts.clean.txt", "r", encoding="latin1") as f:
#     texts = f.read().strip().split("\n")
BoW = np.load(f"temp/{dataset}.BoW.npy")
word2index = joblib.load(f"temp/{dataset}.word2index.pkl")
dic = Dictionary()
dic.token2id = word2index
vectorizer = joblib.load(f"temp/{dataset}.vectorizer.pkl")
vocab = pd.DataFrame(columns=['word', 'index'])
vocab['word'] = vectorizer.get_feature_names()
vocab['index'] = vocab.index

BoW = torch.from_numpy(BoW)
BoW = BoW.to(torch.float32)
labels = np.load(f"temp/{dataset}.targets.npy")

BoW_train, BoW_test, labels_train, labels_test = train_test_split(BoW, labels, test_size=0.2, random_state=0)

train_dataset = BoWDataset(BoW_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = BoWDataset(BoW_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# model = NVDM(BoW.shape[1], num_topics=n_topic)
model = ProdLDA(BoW.shape[1], num_topics=n_topic)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

model.to(device)
def train():
    for epoch in range(epochs):
        ppx_sum = 0
        kld_sum = 0
        word_count = 0
        doc_count = 0
        loss_sum = 0
        diversity = []
        for data, _, count_batch in train_loader:
            word_count += torch.sum(count_batch)
            
            data = data.cuda()
            
            sample, logits, kld, rec_loss = model(data)
            loss = kld + rec_loss
            
            loss_sum += torch.sum(loss)
            kld_sum += torch.mean(kld)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            count_batch = torch.add(count_batch, 1e-12)
            ppx_sum += torch.sum(torch.div(loss.cpu(), count_batch))
            doc_count += len(data)

            beta = model.dec_projection.weight.cpu().detach().T
            diversity.append(evaluate(beta, data, dic))
        print_ppx = torch.exp(loss_sum / word_count)
        print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(train_loader)
        print('| Epoch train: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld),
              '| Loss: {:.5}'.format(loss_sum),
              '| Diversity: {:.5}'.format(sum(diversity)/len(diversity)))

def test():
    ppx_sum = 0
    kld_sum = 0
    word_count = 0
    doc_count = 0
    loss_sum = 0
    diversity = []
    for data, _, count_batch in test_loader:
        word_count += torch.sum(count_batch)
        
        data = data.cuda()
        
        sample, logits, kld, rec_loss = model(data)
        loss = kld + rec_loss
        
        loss_sum += torch.sum(loss)
        kld_sum += torch.mean(kld)
        # count_batch += 1e-12
        ppx_sum += torch.sum(torch.div(loss.cpu(), count_batch))
        doc_count += len(data)

        beta = model.dec_projection.weight.cpu().detach().T
        diversity.append(evaluate(beta, data, dic))
    print_ppx = torch.exp(loss_sum / word_count)
    print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
    print_kld = kld_sum / len(test_loader)
    print('| Epoch test:',
          '| Perplexity: {:.9f}'.format(print_ppx),
          '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
          '| KLD: {:.5}'.format(print_kld),
          '| Loss: {:.5}'.format(loss_sum),
          '| Diversity: {:.5}'.format(sum(diversity)/len(diversity)))
          
def generate_train_repr():
    samples = []
    with torch.no_grad():
        for data, _, _ in train_loader:

            data = data.cuda()

            sample, logits, kld, rec_loss = model(data)

            samples.append(sample)

    train_repr = torch.cat(samples, dim=0).cpu().numpy()
    return train_repr

        
def generate_test_repr():
    samples = []
    with torch.no_grad():
        for data, _, _ in test_loader:

            data = data.cuda()

            sample, logits, kld, rec_loss = model(data)

            samples.append(sample)

    test_repr = torch.cat(samples, dim=0).cpu().numpy()
    return test_repr

def plot_word_cloud(b, ax, vocab, n):
    sorted_, indices = torch.sort(b, descending=True)
    # df = pd.DataFrame(indices[:100].cpu().numpy(), columns=['index'])
    # words = pd.merge(df, vocab[['index', 'word']],
    #                  how='left', on='index')['word'].values.tolist()
    id2token = dic

    words= [id2token[idx] for idx in indices[:100].cpu().detach().numpy().tolist()]

    sizes = (sorted_[:100] * 1000).cpu().detach().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}

    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")

if __name__ == "__main__":
    train()
    test()
    # beta = model.dec_projection.weight.cpu().detach().T
    
    # fig, axs = plt.subplots(7, 3, figsize=(14, 24))
    # for n in range(beta.shape[0]):
        # i, j = divmod(n, 3)
        # plot_word_cloud(beta[n], axs[i, j], vocab, n)
    # axs[-1, -1].axis('off');
    # plt.show()
    # generate topic distributions used to classification

    # train_repr = generate_train_repr()
    # test_repr = generate_test_repr()

    # clf = svm.SVC(kernel='rbf')
    #
    # clf.fit(train_repr, labels_train)
    
    # print("Accuracy on training set:", clf.score(train_repr, labels_train))
    # print("Accuracy on testing set:", clf.score(test_repr, labels_test))
