#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = pd.read_csv('hotel.csv') 
print(data.shape)
reviews = data['reviews']
print(reviews.shape)


# In[3]:


validationdata = pd.read_csv('validation.csv') 
print(validationdata.shape)
validationreviews = validationdata['reviews']
print(validationreviews.shape)


# In[4]:


traindata = pd.read_csv('train.csv') 
print(traindata.shape)
trainreviews = traindata['reviews']
print(trainreviews.shape)


# In[5]:


testdata = pd.read_csv('test.csv') 
print(testdata.shape)
testreviews = testdata['reviews']
print(testreviews.shape)


# In[6]:


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# In[7]:


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[8]:


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


# In[9]:


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


# In[10]:


max_seq_len = 200
embed_dim = 300
hidden_dim = 200
polarities_dim = 2
glove_fname = 'glove.42B.300d.txt' 


# In[11]:


def build_tokenizer(dataset, max_seq_len):
    tokenizer = Tokenizer(max_seq_len)
    for data in dataset:
        tokenizer.fit_on_text(data)
    return tokenizer


# In[12]:


tokenizer = build_tokenizer(dataset = reviews, max_seq_len = max_seq_len)


# In[13]:


word2idx = tokenizer.word2idx


# In[14]:


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


# In[15]:


word_vec = load_word_vec(glove_fname, word2idx)


# In[16]:


def build_embedding_matrix(word2idx, word_vec, embed_dim):
    embedding_matrix = np.zeros((len(word2idx)+2, embed_dim))
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = vec
    return embedding_matrix


# In[17]:


embedding_matrix = build_embedding_matrix(word2idx, word_vec, embed_dim)


# In[18]:


class CreateDataset(Dataset):
    def __init__(self, dataset, sentence_tokenizer):
        
        all_data = []
        reviews = dataset['reviews']
        aspect1_ratings = dataset['aspects1_rating']
        aspect2_ratings = dataset['aspects2_rating']
        aspect3_ratings = dataset['aspects3_rating']
        aspect4_ratings = dataset['aspects4_rating']
        overall_ratings = dataset['rating']
        
        for i in range(len(reviews)):
            
            review = reviews[i]
            review_token = preprocess(review)
            review_indices = sentence_tokenizer.text_to_sequence(review)
            
            aspect1_rating = self.polarity(aspect1_ratings[i])
            aspect2_rating = self.polarity(aspect2_ratings[i])
            aspect3_rating = self.polarity(aspect3_ratings[i])
            aspect4_rating = self.polarity(aspect4_ratings[i])
            
            overall_rating = self.polarity(overall_ratings[i])
            
            words_pos = self.get_absolute_pos(review_indices)
        
            newdata = {
                'review': review,
                'review_token': review_token,
                'review_indices': review_indices, 
            
                'aspect1_rating': aspect1_rating,
                'aspect2_rating': aspect2_rating,
                'aspect3_rating': aspect3_rating,
                'aspect4_rating': aspect4_rating,
                
                'overall_rating': overall_rating,
                
                'words_pos': words_pos
                }
            
            all_data.append(newdata)

        self.data = all_data
    
    def polarity(self, data):
        if(data>3):
            newdata = 1
        else:
            newdata = 0
        return newdata

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def get_absolute_pos(self,review_indices):
        # batch = []
        start_idx = 1
        word_pos = []
        for pos in review_indices:
            if int(pos) == 0:
                word_pos.append(0)
            else:
                word_pos.append(start_idx)
                start_idx+=1
        batch = torch.LongTensor(word_pos)
        return batch


# In[19]:


trainset = CreateDataset(traindata,tokenizer)
validationset = CreateDataset(validationdata,tokenizer)
testset = CreateDataset(testdata,tokenizer)


# In[20]:


batch_size = 64


# In[21]:


train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_data_loader = DataLoader(dataset=validationset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, drop_last=True)


# In[22]:


class DynamicLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        
        self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  


    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        out_pack, (ht, ct) = self.lstm(x_emb_p, None)
   
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  
            out = out[x_unsort_idx]
            
            """unsort: out c"""

            ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


# In[23]:


class MAN(nn.Module):
    def __init__(self,embedding_matrix, embed_dim, hidden_dim, polarities_dim, max_seq_len, device):
        
        super(MAN, self).__init__()
        
        # Embedding   
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.position_embed = nn.Embedding(max_seq_len+1, embed_dim, padding_idx=0)
        
        # RNN
        self.lstm = DynamicLSTM(embed_dim*2, hidden_dim, num_layers=1, batch_first=True, bidirectional=False, dropout=0)
        
        # Attention
        self.weight_a1_alpha = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a1_alpha = nn.Parameter(torch.Tensor(1))
    
        self.weight_a2_alpha = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a2_alpha = nn.Parameter(torch.Tensor(1))
        
        self.weight_a3_alpha = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a3_alpha = nn.Parameter(torch.Tensor(1))
        
        self.weight_a4_alpha = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a4_alpha = nn.Parameter(torch.Tensor(1))
    
        self.weight_a1_gamma = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a1_gamma = nn.Parameter(torch.Tensor(1))
        
        self.weight_a2_gamma = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a2_gamma = nn.Parameter(torch.Tensor(1))
        
        self.weight_a3_gamma = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a3_gamma = nn.Parameter(torch.Tensor(1))
        
        self.weight_a4_gamma = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_a4_gamma = nn.Parameter(torch.Tensor(1))
        
        # Aspects Logits
        self.dense1 = nn.Linear(hidden_dim, polarities_dim)
        self.dense2 = nn.Linear(hidden_dim, polarities_dim)
        self.dense3 = nn.Linear(hidden_dim, polarities_dim)
        self.dense4 = nn.Linear(hidden_dim, polarities_dim)
        
        # Overall Logits
        self.dense = nn.Linear(hidden_dim*4, polarities_dim)
        
        # Identity Matirx 
        self.identity = torch.eye(hidden_dim, hidden_dim).cuda()
      
    def forward(self, review_indices, words_pos):
        
        # Embedding 
        text_raw_indices = review_indices        # batch_size x seq_len
        text_emb = self.embed(text_raw_indices)  # batch_size x seq_len x embed_dim
        position_emb = self.position_embed(words_pos)  # batch_size x seq_len x embed_dim
        
        text_len = torch.sum(text_raw_indices != 0, dim=-1)
        text_pos = torch.cat((position_emb,text_emb), dim=-1)
        
        # lstm
        lstmout, _ = self.lstm(text_pos, text_len)
        lstmout_pool = torch.unsqueeze(torch.div(torch.sum(lstmout, dim=1), text_len.float().view(text_len.size(0), 1)), dim=1)
        
        # aspect1 
        # Self-Attention
        alpha1 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout, self.weight_a1_alpha), torch.transpose(lstmout, 1, 2)), self.bias_a1_alpha)), dim=1)
        aspect1_output = torch.bmm(alpha1, lstmout)
        # Position-awar Attention
        gamma1 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout_pool, self.weight_a1_gamma), torch.transpose(lstmout, 1, 2)), self.bias_a1_gamma)), dim=1)
        aspect1_output = torch.squeeze(torch.bmm(gamma1, aspect1_output), dim=1)
        
        # aspect2
        # Self-Attention
        alpha2 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout, self.weight_a2_alpha), torch.transpose(lstmout, 1, 2)), self.bias_a2_alpha)), dim=1)
        aspect2_output = torch.bmm(alpha2, lstmout)
        # Position-awar Attention
        gamma2 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout_pool, self.weight_a2_gamma), torch.transpose(lstmout, 1, 2)), self.bias_a2_gamma)), dim=1)
        aspect2_output = torch.squeeze(torch.bmm(gamma2, aspect2_output), dim=1)
        
        # aspect3
        # Self-Attention
        alpha3 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout, self.weight_a3_alpha), torch.transpose(lstmout, 1, 2)), self.bias_a3_alpha)), dim=1)
        aspect3_output  = torch.bmm(alpha3, lstmout)
        # Position-awar Attention
        gamma3 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout_pool, self.weight_a3_gamma), torch.transpose(lstmout, 1, 2)), self.bias_a3_gamma)), dim=1)
        aspect3_output = torch.squeeze(torch.bmm(gamma3, aspect3_output), dim=1)
        
        # aspect4
        # Self-Attention
        alpha4 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout, self.weight_a4_alpha), torch.transpose(lstmout, 1, 2)), self.bias_a4_alpha)), dim=1)
        aspect4_output = torch.bmm(alpha4, lstmout)
        # Position-awar Attention
        gamma4 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(lstmout_pool, self.weight_a4_gamma), torch.transpose(lstmout, 1, 2)), self.bias_a4_gamma)), dim=1)
        aspect4_output = torch.squeeze(torch.bmm(gamma4, aspect4_output), dim=1)

        orthogonal_a = torch.cat((alpha1, alpha2, alpha3, alpha4), dim=1)
        orthogonal_g = torch.cat((gamma1, gamma2, gamma3, gamma4), dim=1)
        
        orthogonal_a = torch.bmm(orthogonal_a.permute(0, 2, 1), orthogonal_a) 
        orthogonal_b = torch.bmm(orthogonal_g.permute(0, 2, 1), orthogonal_g) 
        
        orthogonal_a = orthogonal_a - self.identity[:orthogonal_a.size(1),:orthogonal_a.size(1)].expand(orthogonal_a.size(0), -1, -1)
        orthogonal_b = orthogonal_b - self.identity[:orthogonal_b.size(1),:orthogonal_b.size(1)].expand(orthogonal_b.size(0), -1, -1)
        
        orthogonal_a_loss = torch.norm(orthogonal_a) / orthogonal_a.size(1)
        orthogonal_b_loss = torch.norm(orthogonal_b) / orthogonal_b.size(1)
        
        # Aspects logits
        aspect1_rating = self.dense1(aspect1_output)
        aspect2_rating = self.dense2(aspect2_output)
        aspect3_rating = self.dense3(aspect3_output)
        aspect4_rating = self.dense4(aspect4_output)  
        
        # Overall logits
        overall_output  = torch.cat((aspect1_output,aspect2_output,aspect3_output,aspect4_output),1)  
        overall_rating  = self.dense(overall_output)
    
        return overall_rating,aspect1_rating,aspect2_rating,aspect3_rating,aspect4_rating,orthogonal_a_loss, orthogonal_b_loss, alpha1, alpha2


# In[24]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[25]:


mymodel = MAN(embedding_matrix, embed_dim, hidden_dim, polarities_dim, max_seq_len, device)
mymodel = mymodel.to(device)
mymodel


# In[26]:


# function to reset parameters
def reset_params(mymodel):
    for child in mymodel.children():
        for p in child.parameters():
            if p.requires_grad:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('Finished')


# In[27]:


# function to show parameters
def paramsshow(net):
    print(net)
    params = list(net.parameters())
    print("lenghth of parameters:",len(params))
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size()) 


# In[28]:


# paramsshow(mymodel)


# In[29]:


criterion = nn.CrossEntropyLoss()
logdir = 'log'
learning_rate = 0.005
l2reg = 0.01
log_step = 10
lambbda = 0.5


# In[30]:


params = filter(lambda p: p.requires_grad, mymodel.parameters())


# In[31]:


optimizer = torch.optim.Adam(params, lr= learning_rate, weight_decay=l2reg)


# In[32]:


num_epoch = 20


# In[33]:


reset_params(mymodel)


# In[34]:


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
          
            optimizer.zero_grad()      
           
            review_indices = sample_batched['review_indices'].to(device)
            
            rating = sample_batched['overall_rating'].to(device)
            value_rating = sample_batched['aspect1_rating'].to(device)
            atmosphere_rating = sample_batched['aspect2_rating'].to(device)
            service_rating = sample_batched['aspect3_rating'].to(device)
            food_rating = sample_batched['aspect4_rating'].to(device)
            
            words_pos = sample_batched['words_pos'].to(device)
            
            overall_rating,aspect1_rating,aspect2_rating,aspect3_rating,aspect4_rating,orthogonal_a_loss, orthogonal_b_loss, alpha1, alpha2 = model(review_indices, words_pos) 
            
            loss1 = criterion(overall_rating, rating)
            
            loss2 = criterion(aspect1_rating, value_rating)
            losses.append(loss2)
            loss3 = criterion(aspect2_rating, atmosphere_rating)
            losses.append(loss3)
            loss4 = criterion(aspect3_rating, service_rating)
            losses.append(loss4)
            loss5 = criterion(aspect4_rating, food_rating)
            losses.append(loss5)
           
            min_loss = heapq.nsmallest(4, losses)
            
            com_loss = min_loss[0] + min_loss[1] + min_loss[2] + min_loss[3]
  
            loss = loss1 + lambbda * orthogonal_a_loss + lambbda * orthogonal_b_loss + lambbda * com_loss 
    
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
        
    torch.save(mymodel, 'HotelMAN-LSTM.pkl')
    # torch.save(mymodel.state_dict(), 'HotelParam.pkl')
    # torch.save(mymodel.state_dict(), 'params.pth')
    print('#########Finished Training#########')

    return accuracy_point,alpha1, alpha2


# In[35]:


accuracy_point, alpha1, alpha2 = train(mymodel, train_data_loader, criterion, optimizer, log_step, num_epoch, lambbda, device)


# In[36]:


def evaluate_acc_f1(test_data_loader,model):
    n_test_correct, n_test_total = 0, 0
    n_test_correct1,n_test_correct2,n_test_correct3,n_test_correct4 = 0,0,0,0

    t_targets_all, t_outputs_all = None, None
    
    t_targets_all1, t_outputs_all1 = None, None
    t_targets_all2, t_outputs_all2 = None, None
    t_targets_all3, t_outputs_all3 = None, None
    t_targets_all4, t_outputs_all4 = None, None
    
    # switch model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(test_data_loader):
            
            t_sentence = t_sample_batched['review_indices'].to(device)
  
            t_targets = t_sample_batched['overall_rating'].to(device)
       
            t_targets1 = t_sample_batched['aspect1_rating'].to(device)
            t_targets2 = t_sample_batched['aspect2_rating'].to(device)
            t_targets3 = t_sample_batched['aspect3_rating'].to(device)
            t_targets4 = t_sample_batched['aspect4_rating'].to(device)
            
            words_pos = t_sample_batched['words_pos'].to(device)
            
            t_overall,foodrating,valuerating,servicerating,atmosphererating,orthogonal_a_loss,orthogonal_b_loss = model(t_sentence, words_pos) 
        
            n_test_correct += (torch.argmax(t_overall, -1) == t_targets).sum().item()
            
            n_test_correct1 += (torch.argmax(foodrating, -1) == t_targets1).sum().item()
            n_test_correct2 += (torch.argmax(valuerating, -1) == t_targets2).sum().item()
            n_test_correct3 += (torch.argmax(servicerating, -1) == t_targets3).sum().item()
            n_test_correct4 += (torch.argmax(atmosphererating, -1) == t_targets4).sum().item()
            
            n_test_total += len(t_overall)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_overall   
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_overall), dim=0)
                
                
            if t_targets_all1 is None:      
                t_targets_all1 = t_targets1
                t_outputs_all1 = foodrating
            else:
                t_targets_all1 = torch.cat((t_targets_all1, t_targets1), dim=0)
                t_outputs_all1 = torch.cat((t_outputs_all1, foodrating), dim=0)    
                
            if t_targets_all2 is None:      
                t_targets_all2 = t_targets2
                t_outputs_all2 = valuerating
            else:
                t_targets_all2 = torch.cat((t_targets_all2, t_targets2), dim=0)
                t_outputs_all2 = torch.cat((t_outputs_all2, valuerating), dim=0) 
                
            if t_targets_all3 is None:      
                t_targets_all3 = t_targets3
                t_outputs_all3 = servicerating
            else:
                t_targets_all3 = torch.cat((t_targets_all3, t_targets3), dim=0)
                t_outputs_all3 = torch.cat((t_outputs_all3, servicerating), dim=0) 
                
            if t_targets_all4 is None:      
                t_targets_all4 = t_targets4
                t_outputs_all4 = atmosphererating
            else:
                t_targets_all4 = torch.cat((t_targets_all4, t_targets4), dim=0)
                t_outputs_all4 = torch.cat((t_outputs_all4, atmosphererating), dim=0) 
                
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
        print ('test_acc: %.4f, f1: %.4f' %(test_acc,f1))
        
        test_acc1 = n_test_correct1 / n_test_total
        f11 = metrics.f1_score(t_targets_all1.cpu(), torch.argmax(t_outputs_all1, -1).cpu(), labels=[0, 1], average='macro')
        print ('test_acc1: %.4f, f11: %.4f' %(test_acc1,f11))
        
        test_acc2 = n_test_correct2 / n_test_total
        f12 = metrics.f1_score(t_targets_all2.cpu(), torch.argmax(t_outputs_all2, -1).cpu(), labels=[0, 1], average='macro')
        print ('test_acc2: %.4f, f12: %.4f' %(test_acc2,f12))
        
        test_acc3 = n_test_correct3 / n_test_total
        f13 = metrics.f1_score(t_targets_all3.cpu(), torch.argmax(t_outputs_all3, -1).cpu(), labels=[0, 1], average='macro')
        print ('test_acc3: %.4f, f13: %.4f' %(test_acc3,f13))
        
        test_acc4 = n_test_correct4 / n_test_total
        f14 = metrics.f1_score(t_targets_all4.cpu(), torch.argmax(t_outputs_all4, -1).cpu(), labels=[0, 1], average='macro')
        print ('test_acc4: %.4f, f14: %.4f' %(test_acc4,f14))


# In[37]:


newmodel = torch.load('HotelMAN-LSTM.pkl')


# In[38]:


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




