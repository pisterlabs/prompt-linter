import csv
import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
from sklearn.model_selection import train_test_split #conda install -c anaconda scikit-learn
import datetime
import matplotlib.pyplot as plt #conda install -c conda-forge matplotlib
import time
from pprint import pprint
import os
import math

import spacy  # conda install -c conda-forge spacy
import nltk #conda install -c anaconda nltk
import string
from nltk.corpus import stopwords
import preprocessor as p  #pip install tweet-preprocessor
import logging  # Setting up the loggings to monitor gensim
from nltk.stem import WordNetLemmatizer
from string import punctuation as punc

import gensim #conda install -c anaconda gensim
import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.models as gsm
from gensim.test.utils import datapath

import regex
import emoji #pip install emoji --upgrade #conda install -c conda-forge emoji


import torch #conda install -c pytorch pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
#import torchvision.transforms as transforms #conda install -c pytorch torchvision
from torch.autograd import Variable

####BERT

from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)


nlp = spacy.load('en_core_web_sm')#python -m spacy download en_core_web_sm

torch.manual_seed(1)




# df = pd.read_csv( 'data/climate_ad_gt.csv',low_memory=False)  # (25054, 20)
# #df = pd.read_csv( 'data/unq_climate_ad_gt.csv',low_memory=False)  # (5414, 20)
# #print(df) #[25054 rows x 20 columns]

df = pd.read_csv( 'data/finetune_score.csv',low_memory=False)  # (25054, 20)
print(df.columns) #[25054 rows x 20 columns]


### we need to split train-val-test beased on unique funding entity
unq_fe = df['funding_entity'].unique()
df_fe = pd.DataFrame(unq_fe, columns=['funding_entity']) #408 unq fe
#print(df_fe) #[408 rows x 1 columns]

# print(df.shape) #(25054, 20)
#
df_train_val_fe=df_fe.sample(frac=0.8,random_state=42) #random state is a seed value (fold1 = 22, fold2 = 1, fold3 = 42 )
df_test_fe=df_fe.drop(df_train_val_fe.index)


df_train_fe =df_train_val_fe.sample(frac=0.8,random_state=42)
df_valid_fe = df_train_val_fe.drop(df_train_fe.index)

df_train_fe = df_train_fe.reset_index(drop=True)
df_valid_fe = df_valid_fe.reset_index(drop=True)
df_test_fe = df_test_fe.reset_index(drop=True)

#print(df_train_fe.shape, df_valid_fe.shape, df_test_fe.shape ) #(261, 1) (65, 1) (82, 1)

## get rows from a DataFrame that are in another DataFrame in Python 

df_train = df.merge(df_train_fe, on= "funding_entity")
df_train = df_train.reset_index(drop=True)
#print(df_train) ##[17018 rows x 20 columns]


df_valid = df.merge(df_valid_fe, on= "funding_entity")
df_valid = df_valid.reset_index(drop=True)
#print(df_valid) ##[2826 rows x 20 columns]

df_test = df.merge(df_test_fe, on= "funding_entity")
df_test = df_test.reset_index(drop=True)
#print(df_test) ##[5210 rows x 20 columns]
#sys.exit()

#print(df_train['stance'].unique(), df_valid['stance'].unique(), df_test['stance'].unique())

### Replace a single value with a new value for an individual DataFrame column:

df_train['stance'] = df_train['stance'].replace(-1,2)
df_valid['stance'] = df_valid['stance'].replace(-1,2)
df_test['stance'] = df_test['stance'].replace(-1,2)
#print(df_train['stance'].unique(), df_valid['stance'].unique(), df_test['stance'].unique())

# sys.exit()

# df_train['stance'] = df_train['stance'].fillna('')
# df_valid['stance'] = df_valid['stance'].fillna('')
# df_test['stance'] = df_test['stance'].fillna('')


# print(df_train['stance'].unique(), df_valid['stance'].unique(), df_test['stance'].unique())


# sys.exit()

# print(df_train)
# print(df_valid)
# print(df_test)

print(df_train.shape, df_valid.shape,  df_test.shape) #(8874, 21) (2219, 21) (2773, 21)
#print(df.columns)
#sys.exit()

sentences_train = df_train.ad_creative_body
sentences_valid = df_valid.ad_creative_body
sentences_test = df_test.ad_creative_body


# Get the lists of sentences and their labels.
            
            
sentences_train = np.array(sentences_train)
# labels_train = df_train.stance.values
# print(type(labels_train[0]))
labels_train = []
for i in range (0,df_train.shape[0]):
    # labels_train.append(int(df_train.stance[i]))
    labels_train.append(int(df_train.stance[i]))
#print(type(labels_train[0]))
print(type(sentences_train))

sentences_valid = np.array(sentences_valid)
#labels_valid = df_valid.stance.values
labels_valid = []
for i in range (0,df_valid.shape[0]):
    labels_valid.append(int(df_valid.stance[i]))
print(type(sentences_valid))

sentences_test = np.array(sentences_test)
labels_test = df_test.stance.values
print(type(sentences_test))


prob_train = [str(item) for item in df_train.probability] 
prob_valid = [str(item) for item in df_valid.probability] 
prob_test = [str(item) for item in df_test.probability] 
#print(type(sentences_test))
#list_of_strings = [str(item) for item in list_of_floats]  #convert list of floats to list of strings



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


#The first four features are in tokenizer.encode, but I’m using tokenizer.encode_plus to get the fifth item (attention masks).

def token_id(sentences_train):
    input_ids_train = []
    attention_masks_train = []

    # For every sentence...
    for sent in sentences_train:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 100,           # Pad & truncate all sentences.
                            #padding = 'max_length',
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation=True, #explicitely truncate examples to max length. #my add
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

                        

        # Add the encoded sentence to the list.    
        input_ids_train.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_train.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    return input_ids_train, attention_masks_train
## ad text
input_ids_train_ad, attention_masks_train_ad = token_id(sentences_train)
input_ids_valid_ad, attention_masks_valid_ad = token_id(sentences_valid )
input_ids_test_ad, attention_masks_test_ad = token_id(sentences_test)

def token_id_thm(sentences_train):
    input_ids_train = []
    attention_masks_train = []

    # For every sentence...
    for sent in sentences_train:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        # encoded_dict = tokenizer.encode_plus(
        #                     sent,                      # Sentence to encode.
        #                     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        #                     max_length = 10,           # Pad & truncate all sentences.
        #                     #padding = 'max_length',
        #                     pad_to_max_length = True,
        #                     return_attention_mask = True,   # Construct attn. masks.
        #                     truncation=True, #explicitely truncate examples to max length. #my add
        #                     return_tensors = 'pt',     # Return pytorch tensors.
        #                )
        nt = sent + " [SEP]"
        encoded_dict = tokenizer.encode_plus(
                            nt,                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            max_length = 10,           # Pad & truncate all sentences.
                            #padding = 'max_length',
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation=True, #explicitely truncate examples to max length. #my add
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
                        

        # Add the encoded sentence to the list.    
        input_ids_train.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_train.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    return input_ids_train, attention_masks_train

##thm prob
input_ids_train_thm, attention_masks_train_thm = token_id_thm(prob_train)
input_ids_valid_thm, attention_masks_valid_thm = token_id_thm(prob_valid )
input_ids_test_thm, attention_masks_test_thm = token_id_thm(prob_test)

##labels
labels_train = torch.tensor(labels_train)
labels_valid = torch.tensor(labels_valid)
labels_test = torch.tensor(labels_test)

###concat ads + thm prob
input_ids_train = torch.cat((input_ids_train_ad, input_ids_train_thm), 1) #concat with dim= 1
input_ids_valid = torch.cat((input_ids_valid_ad, input_ids_valid_thm), 1) #concat with dim= 1
input_ids_test = torch.cat((input_ids_test_ad, input_ids_test_thm), 1) #concat with dim= 1


attention_masks_train = torch.cat((attention_masks_train_ad, attention_masks_train_thm), 1) #concat with dim= 1
attention_masks_valid = torch.cat((attention_masks_valid_ad, attention_masks_valid_thm), 1) #concat with dim= 1
attention_masks_test = torch.cat((attention_masks_test_ad, attention_masks_test_thm), 1) #concat with dim= 1

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
val_dataset = TensorDataset(input_ids_valid, attention_masks_valid, labels_valid)
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

import torch
# torch.cuda.empty_cache() 
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    #device = torch.device("cuda") # select the zeroth GPU with this line: gpu = 0
    #device = torch.cuda.set_device(1)  #wrong provide device = None  
    device = torch.device(0) #(use cuda device 1) for gpu = 1
    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name())

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#device = torch.device("cpu") # uncomment for cpu use


#print ('Current cuda device ', torch.cuda.current_device())
print ('Current cuda device ', device)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


#Train Our Classification Model. BertForSequenceClassification

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 3, # The number of output labels--3 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
#model.cuda() #comment it if you use cpu only
#print ('Current cuda device ', torch.cuda.current_device())
#model.to(torch.cuda.current_device())
model.to(device)



# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 

# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                 )

# ##Hyper1
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                 )
# ##Hyper2
# optimizer = AdamW(model.parameters(),
#                   lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.01
#                 )

# #Hyper3
# optimizer = AdamW(model.parameters(),
#                   lr = 1e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.001
#                 )

##Hyper4
optimizer = AdamW(model.parameters(),
                  lr = 1e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                  weight_decay = 0.01
                )

# ##Hyper5
# optimizer = AdamW(model.parameters(),
#                   lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.001
#                 )


# ##Hyper6
# optimizer = AdamW(model.parameters(),
#                   lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.001
#                 )

# ##Hyper7
# optimizer = AdamW(model.parameters(),
#                   lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.01
#                 )


##Hyper8
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.1
#                 )

# ##Hyper9
# optimizer = AdamW(model.parameters(),
#                   lr = 1e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.0001
#                 )

##Hyper10
# optimizer = AdamW(model.parameters(),
#                   lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
#                   weight_decay = 0.0001
#                 )

## Define a helper function for calculating accuracy.


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds.cpu(), axis=1).flatten()
    #print("pred_flat", pred_flat.numpy())
    #print("labels", labels)
    labels_flat = labels.flatten()
    #print("labels_flat", labels_flat)
    #print("np.sum(pred_flat == labels_flat) / len(labels_flat)", np.sum(pred_flat.numpy() == labels_flat) / len(labels_flat))
    return np.sum(pred_flat.numpy() == labels_flat) / len(labels_flat)

## Helper function for formatting elapsed times as hh:mm:ss
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



###Training Loop
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []


def uniform(state_dicts, state_dicts_to_use):
    with torch.no_grad():
        merged = {}
        for key in state_dicts[0]:
            if state_dicts[0][key].dtype == torch.int64 or state_dicts[0][key].dtype == torch.uint8:
                merged[key] = state_dicts[0][key]
            else:
                merged[key] = torch.mean(torch.stack(
                    [sd[key] for idx, sd in enumerate(state_dicts) if idx in state_dicts_to_use]), axis=0)
        return merged

def validation_perform(model):
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.

    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            # (loss, logits) = model(b_input_ids, 
            #                        token_type_ids=None, 
            #                        attention_mask=b_input_mask,
            #                        labels=b_labels)
            outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
            
        # Accumulate the validation loss.
        total_eval_loss += outputs.loss.item()

        # Move logits and labels to CPU
        logits = outputs.logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        label_ids = b_labels.detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(outputs.logits, label_ids)
        

    # Report the final accuracy for this validation run.
    #print("total_eval_accuracy", total_eval_accuracy) 
    #print("len(validation_dataloader)", len(validation_dataloader)) 
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(" Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
    #sys.exit()

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    return avg_val_accuracy


def greedy_soup(state_dicts):
    # Note: state_dicts should be sorted in decreasing order of validation accuracy
    state_dicts_to_use = []
    current_val_acc = 0.0
    for i in range(len(state_dicts)):
        new_state_dict = uniform(state_dicts, state_dicts_to_use + [i])
        model.load_state_dict(new_state_dict)
        val_acc = validation_perform(model)
        if val_acc >= current_val_acc:
            current_val_acc = val_acc
            state_dicts_to_use.append(i)
    return uniform(state_dicts, state_dicts_to_use)

device = torch.device(0)
state_dicts = []

for f in os.listdir('output_point_estimate_thm/'):
    #print(f)
    if f[-2:] == 'pt':
        print(f'Loading {f}')
        state_dicts.append(torch.load(os.path.join('output_point_estimate_thm/', f), map_location=device))

###save results from the validation set based on best model from each heyper parameter
val_results = []
new_state_dicts = []
for i in range (0, len(state_dicts)):
    model.load_state_dict(state_dicts[i])
    val_acc = validation_perform(model)
    val_results.append(val_acc)
    new_state_dicts.append(state_dicts[i])
print(val_results, len(val_results))



#Sort an array's rows by another array in Python
val_results = np.array(val_results)
new_state_dicts = np.array(new_state_dicts)
arr1inds = val_results.argsort()
sorted_arr1 = val_results[arr1inds[::-1]]
sorted_arr2 = new_state_dicts[arr1inds[::-1]]

#print(ranked_candidates)

merged = greedy_soup(sorted_arr2)
model.load_state_dict(merged)
model = model.to(device)



#test_results.append(test(model))
torch.save(model.state_dict(), 'output_point_estimate_thm/greedy_soup_model.pt')

### Load the soup model
model.load_state_dict(torch.load("output_point_estimate_thm/greedy_soup_model.pt"))

# For Test the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = 1 # Evaluate with this batch size.
        )


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids_test)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in test_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()
  label_ids = b_labels.detach().cpu().numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')


# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)



df_test['pred_stance'] = flat_predictions


##Baseline Fine-tuned BERT -- accuracy
df_b = df_test[['id', 'ad_creative_body', 'funding_entity', 'pred_theme', 'time', 'imp',
       'spend', 'region', 'demo', 'avg_imp', 'avg_spend', 'pred_stance', 'stance']] #extracted 13 columns
df_b.to_csv("output_point_estimate_thm/test_bert_stance_greedy_soup.csv")

#print(df_b) #29348
df_b1= df_b.dropna()#dropping nan value
#print(df_b1) #385
# df_b1 = df_b1[df_b1.stance != 2] #dropping rows with truth value -1
# df_b1 = df_b1.reset_index(drop = True)
#print(df_b1) #279
count_label = 0

for i in range (0, df_b1.shape[0]):
    #print(df_b1.stance[i][0], type(df_b1.stance[i]))

    if df_b1.pred_stance[i] == int(df_b1.stance[i]): #count label those are same for both ground truth and prediction
        count_label = count_label + 1
        
        
print('count_label', count_label) 
print('accuracy of baseline : ', (count_label/df_b1.shape[0]) ) 


# if unique ads
#accuracy of baseline :  0.8808864265927978 (fold1 seed 22)
#accuracy of baseline :  0.8771929824561403 (fold2 seed 1)
# accuracy of baseline :  0.8864265927977839 (fold3 seed 42)
# avg accuracy of baseline : 0.8815

### colab soup model
## accuracy of baseline :  0.8799630655586335


##Fine-tuned BERT -- Macro avg F1 score
from sklearn.metrics import f1_score
#print(df_b1.stance.values)
new_y = [] # to handle multi-label
for j in range (0, df_b1.shape[0]):
    # if len(df_b1.stance[j]) > 1:
    #     new_y.append(int(df_b1.stance[j]))
    # else:
    new_y.append(int(df_b1.stance[j]))

#print(new_y)
#x = x.long()
print('Macro-avg F1 score of baseline : ', f1_score(new_y, df_b1.pred_stance.values, average='macro'))


# if unique ads
#Macro-avg F1 score of baseline :  0.842622993897554 (fold1 seed 22)
#Macro-avg F1 score of baseline :  0.8485731012897739 (fold2 seed 1)
#Macro-avg F1 score of baseline :  0.8626490851337164 (fold3 seed 42)
# avg Macro-avg F1 score of baseline : 0.8513


### colab soup model
##Macro-avg F1 score of baseline :  0.8655345634010126


##get the prediction for train, val

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = SequentialSampler(train_dataset), # Select batches randomly
            batch_size = 1 # Trains with this batch size.
        )


# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in train_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()
  label_ids = b_labels.detach().cpu().numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('  train  DONE.')


# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

df_train['pred_stance'] = flat_predictions

df_b = df_train[['id', 'ad_creative_body', 'funding_entity', 'pred_theme', 'time', 'imp',
       'spend', 'region', 'demo', 'avg_imp', 'avg_spend', 'pred_stance', 'stance']] #extracted 13 columns
df_b.to_csv("output_point_estimate_thm/train_bert_stance_greedy_soup.csv")

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = 1 # Evaluate with this batch size.
        )


# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in validation_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()
  label_ids = b_labels.detach().cpu().numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('  val  DONE.')


# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

df_valid['pred_stance'] = flat_predictions

df_b = df_valid[['id', 'ad_creative_body', 'funding_entity', 'pred_theme', 'time', 'imp',
       'spend', 'region', 'demo', 'avg_imp', 'avg_spend', 'pred_stance', 'stance']] #extracted 13 columns
df_b.to_csv("output_point_estimate_thm/valid_bert_stance_greedy_soup.csv")



