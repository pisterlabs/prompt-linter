import os
import numpy as np
import matplotlib.pyplot as plt


from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply , Embedding
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K

# from faker import Faker
import random
from tqdm import tqdm
# from babel.dates import format_date
import matplotlib.pyplot as plt

## Gensim 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel 
from gensim.models.ldamodel import LdaModel


import tensorflow as tf
#import tensorflow_addons as tfa
print(tf.__version__)
from sklearn.model_selection import train_test_split
import os
import io
import numpy as np
import re
import unicodedata
import urllib3
import shutil
import zipfile
import itertools
import pickle
from string import digits

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

file_path =  '../dataset/cleaned_opensubtitles' 
os.chdir(file_path)

def preprocess_sentence(sentence):
    
    num_digits= str.maketrans('','', digits)
    
    sentence= sentence.lower()
    sentence= re.sub(" +", " ", sentence)
    sentence= re.sub("'", '', sentence)
    sentence= sentence.translate(num_digits)
    sentence= re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = sentence.rstrip().strip()
    sentence = "<start> " + sentence + " <end>"
    
    return sentence


text =[]
count = 0

for file in os.listdir():
  with open(file ,'r' , encoding='iso-8859-1') as txtfile:
    for line in txtfile.readlines():

      if count == 100000:
        break
      text.append(preprocess_sentence(line))
      count += 1

max_len = 15
trunc_token = 'post'
oov_tok = "<OOV>"
vocab_size = 20000

tokenizer = Tokenizer(num_words = vocab_size ,oov_token=oov_tok , filters = "" )
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(len(word_index))
sequences = tokenizer.texts_to_sequences(text)
padded = pad_sequences(sequences,maxlen=max_len, truncating=trunc_token,padding = 'post')


d = {}
for index , word in enumerate(word_index.keys()):
  if index + 1 == vocab_size:
    break
  d[word] = index + 1
  
word_index = d

index_word = {}
for word , index in word_index.items():
    index_word[index] = word

os.chdir("../")

a_file = open("../dic/word_index.pkl", "wb")
pickle. dump(word_index, a_file)
a_file. close()

a_file = open("../dic/index_word.pkl", "wb")
pickle. dump(word_index, a_file)
a_file. close()


def create_word_embeddings(file_path):
    
    with open(file_path , 'r') as f:
        wordToEmbedding = {}
        wordToIndex = {}
        indexToWord = {}
        
        for line in f:
            data = line.strip().split()
            token = data[0]
            wordToEmbedding[token] = np.array(data[1:] ,dtype = np.float64)



        
        tokens = sorted(wordToEmbedding.keys())
        for idx , token in enumerate(tokens):
            idx = idx + 1 #for zero masking
            wordToIndex[token] = idx
            indexToWord[idx] = token

    return wordToEmbedding , wordToIndex , indexToWord

wordToEmbedding , wordToIndex , indexToWord = create_word_embeddings('../pretrained_word_embeddings/embedding.txt')



def create_pretrained_embedding_layer(wordToEmbedding , wordToIndex , indexToWord):
    
    vocablen = len(word_index)+1 #for zero masking
    embedding_dimensions = 100
    
    embeddingMatrix = np.zeros((vocablen , embedding_dimensions))
    count = 0
    for word , index in word_index.items():
        if word not in wordToEmbedding.keys():
            embeddingMatrix[index ,:] = np.random.uniform(low = -1 , high =1 ,size = (1,100))
            count +=1
        else :
            
            embeddingMatrix[index , :] = wordToEmbedding[word]
        
    embeddingLayer = Embedding(vocablen , embedding_dimensions , weights = [embeddingMatrix] , trainable = False)
    print(embeddingMatrix.shape)
    print(count)
    
    return embeddingMatrix

embeddingMatrix = create_pretrained_embedding_layer(wordToEmbedding , wordToIndex , indexToWord)

np.save('../dic/embedding.npy', embeddingMatrix)

path = r'../../../lda_checkpoint/topic_model.lda'
lda_model = LdaModel.load(path)

import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
# !python3 -m spacy download en

class LDA(tf.keras.layers.Layer): 
  def __init__(self, lda_model,index_to_word):
    super(LDA, self).__init__(trainable= False, dynamic = True)
    self.lda_model = lda_model
    self.index_to_word = index_to_word

  def build(self, input_shape):
    return

  def gensim_preprocess(self ,lda_model , data):

    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]

    new = []
    for line in data:
      new.append(gensim.utils.simple_preprocess(str(line), deacc=True))

    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

    texts_out = []
    for sent in new:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    
    # print(texts_out)
    corpus = [lda_model.id2word.doc2bow(line) for line in texts_out]

    return corpus

  def get_config(self):
    return {
        'lda_model' : self.lda_model,
        'index_to_word' : self.index_to_word,
    }

  def call(self, inp):
    batch_size , time_steps = inp.shape

    data = []

    for i in range(batch_size):
      line = ""
      for j in range(1,time_steps):
        if inp[i][j].numpy() != 0:

          if index_word[int(inp[i][j].numpy())] == '<end>':
            break;

          line = line + self.index_to_word[int(inp[i][j].numpy())]
        data.append(line)

    data = self.gensim_preprocess(self.lda_model ,data)
    predictions = self.lda_model.get_document_topics(data , minimum_probability = 0.0)
    x = []
    for i in range(batch_size):
      x.append((tf.convert_to_tensor(list(predictions[i]),dtype='float32'))[:,1])
    x = tf.convert_to_tensor(x ,dtype='float32')

    return x
    def compute_output_shape(self, input_shape):
        return (batch_size, 20)

def get_dataset(padded):
    index = 0
    context = np.zeros((50000,max_len) ,dtype='float32')
    response = np.zeros((50000,max_len),dtype= 'float32')
    for idx in range(0,100000,2):
        context[index,:] = padded[idx]
        response[index,:] = padded[idx+1]
        index +=1
    return context , response

context , response = get_dataset(padded)
BUFFER_SIZE = len(context)
BATCH_SIZE = 100
steps_per_epoch = len(context)//BATCH_SIZE
embedding_dim = 100
units = 512
vocab_inp_size = len(word_index) + 1
vocab_tar_size = len(word_index) + 1
# tf.keras.backend.set_floatx('float64')

dataset = tf.data.Dataset.from_tensor_slices((context, response )).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, lda_model , index_word):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim , weights =[embeddingMatrix], trainable=False)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # self.topic_awareness = LDA(lda_model = lda_model , index_to_word = index_word)

  def call(self, x, hidden):
    # topic_vector = self.topic_awareness.call(x)
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE , lda_model , index_word)

class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            # query hidden state shape == (batch_size, hidden size)
            # query_with_time_axis shape == (batch_size, 1, hidden size)
            # values shape == (batch_size, max_len, hidden size)
            # we are doing this to broadcast addition along the time axis to calculate the score
            query_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights

attention_layer = BahdanauAttention(10)

class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim , weights =[embeddingMatrix], trainable=False )
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)

            # used for attention
            self.attention = BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            # enc_output shape == (batch_size, max_length, hidden_size)
            attention_vector, attention_weights = self.attention(hidden, enc_output)

            # context_vector = tf.concat([attention_vector, topic_vector], axis = -1)


            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(attention_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x , initial_state = hidden)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)

            return x, state, attention_weights

decoder = Decoder(vocab_tar_size , embedding_dim, units, BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1 = 0.9,beta_2 = 0.999 , decay = 0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_accuracy = tf.metrics.SparseCategoricalAccuracy()

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = '../checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpoint_dir)))
except:
    print("No checkpoint found at {}".format(checkpoint_dir))

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  accu = 0

  with tf.GradientTape() as tape:
    # print(tf.executing_eagerly())

    enc_output, enc_hidden  = encoder(inp, enc_hidden)

    # print(enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      # print(predictions)
      # print(tf.argmax(predictions,1).shape , targ[:,t].shape)

      loss += loss_function(targ[:, t], predictions)
      train_accuracy.update_state(targ[:,t] , predictions)
      accu += train_accuracy.result()
      # print("accu: ",train_accuracy.result())

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  batch_accu = (accu / int(targ.shape[1]))
 
  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss, batch_accu


import time
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  total_accu = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss , batch_accu = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
    total_accu += batch_accu

    if batch % 2 == 0:
      print('Epoch {} Batch {} Loss {:.4f} Accuracy {:,.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy(),
                                                   batch_accu.numpy()))
  
  print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch,
                                      total_accu / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  if (epoch +1) %10 == 0:

    checkpoint.save(file_prefix = checkpoint_prefix)



checkpoint.save(file_prefix = checkpoint_prefix)
print("checkpoints are saved")
