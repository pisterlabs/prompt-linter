import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import random as rnfv
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torch.nn.utils.rnn import pad_sequence
from torch_snippets import *
from torchmetrics import ConfusionMatrix, F1Score
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import OpenAIGPTModel
from pytorch_pretrained_bert import OpenAIGPTTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('torch version:',torch.__version__)
print('device:', device)
train_data_path = 'threeaggdata/threeaggtrain.csv'
valid_data_path = 'threeaggdata/threeaggtest.csv'
print('Train data path:', train_data_path)
print('Valid data path:', valid_data_path)
batch_size = 8 
word_max_len = 64
h1 = 768
h2 = 128
drop_out_rate = 0.2
epochs = 20
learning_rate = 3e-6
train_df = pd.read_csv(train_data_path, names = ['SOURCE', 'TEXT', 'AGGRESSION_CLASS'], 
                   usecols=['TEXT', 'AGGRESSION_CLASS'])
print('Column of df:', list(train_df))
print('Size of data:', len(train_df))
train_df.head()
valid_df = pd.read_csv(valid_data_path, names = ['SOURCE', 'TEXT', 'AGGRESSION_CLASS'], 
                   usecols=['TEXT', 'AGGRESSION_CLASS'])
print('Column of df:', list(valid_df))
print('Size of data:', len(valid_df))
valid_df.head()

np.random.seed(41)
train_shuffled = train_df.reindex(np.random.permutation(train_df.index))
valid_shuffled = valid_df.reindex(np.random.permutation(valid_df.index))
CAG = train_shuffled[train_shuffled['AGGRESSION_CLASS'] == 'CAG']
OAG = train_shuffled[train_shuffled['AGGRESSION_CLASS'] == 'OAG']
NAG = train_shuffled[train_shuffled['AGGRESSION_CLASS'] == 'NAG']
concated_train = pd.concat([CAG, OAG, NAG], ignore_index=True)
concated_train['LABEL'] = 0
concated_train.loc[concated_train['AGGRESSION_CLASS'] == 'CAG', 'LABEL'] = 0
concated_train.loc[concated_train['AGGRESSION_CLASS'] == 'OAG', 'LABEL'] = 1
concated_train.loc[concated_train['AGGRESSION_CLASS'] == 'NAG', 'LABEL'] = 2
CAG = valid_shuffled[valid_shuffled['AGGRESSION_CLASS'] == 'CAG']
OAG = valid_shuffled[valid_shuffled['AGGRESSION_CLASS'] == 'OAG']
NAG = valid_shuffled[valid_shuffled['AGGRESSION_CLASS'] == 'NAG']
concated_valid = pd.concat([CAG, OAG, NAG], ignore_index=True)
concated_valid['LABEL'] = 0
concated_valid.loc[concated_valid['AGGRESSION_CLASS'] == 'CAG', 'LABEL'] = 0
concated_valid.loc[concated_valid['AGGRESSION_CLASS'] == 'OAG', 'LABEL'] = 1
concated_valid.loc[concated_valid['AGGRESSION_CLASS'] == 'NAG', 'LABEL'] = 2
X_train = concated_train['TEXT']
X_valid = concated_valid['TEXT']
in_features = len(list(X_train))
X_train.head()
class_list = ['CAG', 'OAG', 'NAG']
print('Class list:', class_list)
class_num = len(class_list)
print('Number of class:', class_num)

y_train = to_categorical(concated_train['LABEL'], num_classes=3)
y_valid = to_categorical(concated_valid['LABEL'], num_classes=3)
print('Size of train labels:', y_train.shape)
print('Size of valid labels:', y_valid.shape)
class BertData(Dataset):
    def __init__(self, X, y, word_max_len):
        super().__init__()
        
        self.X = X
        self.y = y
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]'], self.X))
        self.tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), 
                                   maxlen=word_max_len, truncating="post", padding="post", dtype="int")
        self.masks = [[float(i > 0) for i in ii] for ii in self.tokens_ids]
        
        print('Token ids size:', self.tokens_ids.shape)
        print('Masks size:', np.array(self.masks).shape)
        print('y size:', np.array(self.y).shape)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        tokens_id = self.tokens_ids[ind]
        label = self.y[ind]
        mask = self.masks[ind]
        return tokens_id, label, mask
    
    def collate_fn(self, data):
        tokens_ids, labels, masks = zip(*data)
        tokens_ids = torch.tensor(tokens_ids).to(device)
        labels = torch.tensor(labels).float().to(device)
        masks = torch.tensor(masks).to(device)
        return tokens_ids, labels, masks
    
    def choose(self):
        return self[np.random.randint(len(self))]
# Train
train_dataset = BertData(X_train, y_train, word_max_len)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                      collate_fn=train_dataset.collate_fn)

# Valid
validate_dataset = BertData(X_valid, y_valid, word_max_len)
validate_sampler = SequentialSampler(validate_dataset)
validate_dataloader = DataLoader(validate_dataset, sampler=validate_sampler, batch_size=batch_size,
                    collate_fn=validate_dataset.collate_fn)
class Bert_Aggression_Identification_Model(nn.Module):
    def __init__(self, h1, h2, class_num, drop_out_rate):
        super(Bert_Aggression_Identification_Model, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(drop_out_rate)
        self.linear1 = nn.Linear(h1, h2)
        self.linear2 = nn.Linear(h2, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        d = self.dropout(pooled_output)
        x = self.relu(self.linear1(d))
        proba = self.softmax(self.linear2(x))
        
        return proba
mod = Bert_Aggression_Identification_Model(h1,h2,class_num,drop_out_rate).to(device)
def train(data, model, optimizer, loss_fn):
    model.train()
    tokens_ids, labels, masks = data
    outputs = model(tokens_ids, masks)
    loss = loss_fn(outputs, labels)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1)
    
    acc = (sum(preds==labels) / len(labels))
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, acc
@torch.no_grad()
def validate(data, model, loss_fn):
    model.eval()
    tokens_ids, labels, masks = data
    outputs = model(tokens_ids, masks)
    loss = loss_fn(outputs, labels)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1)
    
    acc = (sum(preds==labels) / len(labels))
    
    total_predict.extend(list(preds.cpu().numpy()))
    total_label.extend(list(labels.cpu().numpy()))
    
    return loss, acc
model = Bert_Aggression_Identification_Model(h1, h2, class_num, drop_out_rate).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(epochs):
    n_batch = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        train_loss, train_acc = train(data, model, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print(pos=pos, train_loss=train_loss, 
                   train_acc=train_acc)
        
    total_predict = []
    total_label = []

    n_batch = len(validate_dataloader)
    for i, data in enumerate(validate_dataloader):
        val_loss, val_acc = validate(data, model, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print(pos=pos, val_loss=val_loss, val_acc=val_acc)
    
    scheduler.step()