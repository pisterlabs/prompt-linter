#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import pickle
import math
import spacy
import heapq

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.matutils import softcossim
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

import nltk
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from sklearn import metrics
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertModel


# In[ ]:


data = pd.read_csv('restaurant.csv') 
print(data.shape)
reviews = data['reviews']
print(reviews.shape)


# In[ ]:


validationdata = pd.read_csv('validation.csv') 
print(validationdata.shape)
validationreviews = validationdata['reviews']
print(validationreviews.shape)


# In[ ]:


traindata = pd.read_csv('train.csv') 
print(traindata.shape)
trainreviews = traindata['reviews']
print(trainreviews.shape)


# In[ ]:


testdata = pd.read_csv('test.csv') 
print(testdata.shape)
testreviews = testdata['reviews']
print(testreviews.shape)


# In[ ]:


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# In[ ]:


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[ ]:


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


# In[ ]:


class Tokenizer(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        words = preprocess(text)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        words = preprocess(text)
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


# In[ ]:


max_seq_len = 100
bert_dim = 768 
polarities_dim = 2
dropout = 0.5


# In[ ]:


def build_tokenizer(dataset, max_seq_len):
    tokenizer = Tokenizer(max_seq_len)
    for data in dataset:
        tokenizer.fit_on_text(data)
    return tokenizer


# In[ ]:


tokenizer = build_tokenizer(dataset = reviews, max_seq_len = max_seq_len)


# In[ ]:


word2idx = tokenizer.word2idx


# In[ ]:


def polarity(data):
    if(data>3):
        newdata = 1
    else:
        newdata = 0
    return newdata


# In[ ]:


class CreateDataset(Dataset):
    def __init__(self, dataset, sentence_tokenizer):
        
        all_data = []
        reviews = dataset['reviews']
        value_ratings = dataset['aspects1_rating']
        atmosphere_ratings = dataset['aspects2_rating']
        service_ratings = dataset['aspects3_rating']
        food_ratings = dataset['aspects4_rating']
        ratings = dataset['rating']
        
        for i in range(len(reviews)):
            
            review = reviews[i]
            review_token = preprocess(review)
            review_indices = sentence_tokenizer.text_to_sequence(review)
            
            value_rating = polarity(value_ratings[i])
            atmosphere_rating = polarity(atmosphere_ratings[i])
            service_rating = polarity(service_ratings[i])
            food_rating = polarity(food_ratings[i])
            
            rating = polarity(ratings[i])
            
            newdata = {
                # 'review': review,
                # 'review_token': review_token,
                'review_indices': review_indices, 
                'value_rating': value_rating,
                'atmosphere_rating': atmosphere_rating,
                'service_rating': service_rating,
                'food_rating': food_rating,
                'rating': rating,
                }

            all_data.append(newdata)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# In[ ]:


trainset = CreateDataset(traindata,tokenizer)
validationset = CreateDataset(validationdata,tokenizer)
testset = CreateDataset(testdata,tokenizer)


# In[ ]:


batch_size = 32


# In[ ]:


train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_data_loader = DataLoader(dataset=validationset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, drop_last=True)


# In[ ]:


bert = BertModel.from_pretrained('bert-base-uncased')


# In[ ]:


class ABSABert(nn.Module):
    def __init__(self, bert, bert_dim, polarities_dim, dropout):
        
        super(ABSABert, self).__init__()
        
        
        self.bert = bert
        self.dropout = nn.Dropout(dropout)  
        self.dense = nn.Linear(bert_dim, polarities_dim)
        
    def forward(self, review_indices):
        
        # embedding 
        text_bert_indices = review_indices        # batch_size x seq_len
        _, pooled_output = self.bert(text_bert_indices,output_all_encoded_layers=False)
        
        # text_len = torch.sum(text_bert_indices != 0, dim=-1)
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.dense(pooled_output)
        
        return logits


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


mymodel = ABSABert(bert,bert_dim,polarities_dim,dropout)
mymodel = mymodel.to(device)
mymodel


# In[ ]:


def reset_params(mymodel):
    for child in mymodel.children():
        for p in child.parameters():
            if p.requires_grad:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('Finished')


# In[ ]:


# function to show parameters
def paramsshow(net):
    print(net)
    params = list(net.parameters())
    print("lenghth of parameters:",len(params))
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size()) 


# In[ ]:


# paramsshow(mymodel)


# In[ ]:


criterion = nn.CrossEntropyLoss()
logdir = 'log'
learning_rate = 0.01
l2reg = 0.01
log_step = 10
lambbda = 0.5


# In[ ]:


params = filter(lambda p: p.requires_grad, mymodel.parameters())


# In[ ]:


optimizer = torch.optim.Adam(params, lr= learning_rate, weight_decay=l2reg)


# In[ ]:


num_epoch = 10


# In[ ]:


reset_params(mymodel)


# In[ ]:


def train(model, data_loader, criterion, optimizer, log_step, num_epoch,lambbda, device):
    
    print('#########Start Training#########')
    accuracy_point = []
    global_step = 0
    
    # loop over the dataset multiple times
    for epoch in range(num_epoch):
        
        print('>' * 50)
        print('epoch:', epoch)
        
        running_loss = 0.0
        
        correct = 0
        accuracy = 0
        total = 0
        
        newaccuracy = 0
        newtotal = 0
        newcorrect = 0
        
        # switch model to training mode
        model.train()
        
        for i_batch, sample_batched in enumerate(data_loader):
            
            global_step += 1
            
            losses = []
            # switch model to training mode, clear gradient accumulators
            mymodel.train()
            optimizer.zero_grad()      
           
            review_indices = sample_batched['review_indices'].to(device)
            
            rating = sample_batched['rating'].to(device)

            overall_rating = mymodel(review_indices) 
            
            # losses
            loss = criterion(overall_rating, rating)
            
            loss.backward()
            optimizer.step()
            
            # calculate running loss 
            running_loss += loss.item()
            
            # get accuracy 
            total += rating.size(0)
            correct += (torch.argmax(overall_rating, -1) == rating).sum().item()                       
            newcorrect = correct
            newtotal = total
            newaccuracy = 100*newcorrect/newtotal
            
            if global_step %  log_step == 0:
                
                # print loss and accyracy
                print ('[%2d, %2d] loss: %.3f accuracy: %.2f' %(epoch + 1, i_batch + 1, running_loss,newaccuracy))
                running_loss = 0.0
                newtotal = 0
                newcorrect = 0
                newaccuracy = 0
                       
        accuracy = 100 * correct / total
        print ('epoch: %d, accuracy: %.2f' %(epoch,accuracy))
        accuracy_point.append(accuracy)
        
    torch.save(mymodel, 'RestaurantBert.pkl')
    print('#########Finished Training#########')
    
    return accuracy_point   


# In[ ]:


accuracy_point = train(mymodel, train_data_loader, criterion, optimizer, log_step, num_epoch, lambbda, device)


# In[ ]:


def evaluate_acc_f1(test_data_loader,model):
    n_test_correct, n_test_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    
    # switch model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(test_data_loader):
            t_sentence = t_sample_batched['review_indices'].to(device)
  
            t_targets = t_sample_batched['rating'].to(device)
        
            t_overall = model(t_sentence) 
            
            n_test_correct += (torch.argmax(t_overall, -1) == t_targets).sum().item()
            n_test_total += len(t_overall)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_overall
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_overall), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
    
        print ('test_acc: %.4f, f1: %.4f' %(test_acc,f1))


# In[ ]:


newmodel = torch.load('RestaurantBert.pkl')


# In[ ]:


evaluate_acc_f1(test_data_loader,newmodel)


# In[ ]:


# plot the accuracy of training
def plot_acc(train_accuracy,epoch):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(epoch), train_accuracy, label='Train')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


# In[ ]:


plot_acc(accuracy_point,num_epoch) 


# In[ ]:




