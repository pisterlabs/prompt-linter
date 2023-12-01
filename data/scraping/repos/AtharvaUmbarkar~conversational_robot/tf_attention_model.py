import os
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
import argparse

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply , Embedding
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
# %matplotlib inline

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
import json
import os
import pickle
import io
import re
import unicodedata
import urllib3
import shutil
import zipfile
import itertools
from string import digits

import matplotlib.ticker as ticker

import unicodedata
import time
import speech_recognition as sr

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

max_len = 20
trunc_token = 'post'
oov_tok = "<OOV>"

word_index = pickle.load(open(os.getcwd() + "/dic/word_index.pkl", "rb"))
index_word = pickle.load(open(os.getcwd() + "/dic/index_word.pkl", "rb"))


# word_index = json.load( open( os.getcwd() + "/dic/word_index.json" ) )
# index_word = json.load( open( os.getcwd() + "/dic/index_word.json" ) )

# print(type(word_index))
# print(index_word[10])
# embeddingMatrix

path = os.getcwd() + '/lda_checkpoint/topic_model.lda'
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

BATCH_SIZE = 1
embedding_dim = 50
units = 128
vocab_inp_size = len(word_index)+1
vocab_tar_size = len(word_index)+1
# tf.keras.backend.set_floatx('float64')

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
    self.topic_awareness = LDA(lda_model = lda_model , index_to_word = index_word)

  def call(self, x, hidden):
    topic_vector = self.topic_awareness.call(x)
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state, topic_vector

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

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim , , weights =[embeddingMatrix], trainable=False)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size , activation = "softmax")

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output ,topic_vector):
    # enc_output shape == (batch_size, max_length, hidden_size)
    attention_vector, attention_weights = self.attention(hidden, enc_output)

    context_vector = tf.concat([attention_vector, topic_vector], axis = -1)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x , initial_state = hidden)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size , embedding_dim, units, BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# if not os.path.exists(checkpointdir):
  # os.mkdir(checkpoint_dir)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpoint_dir)))
except:
    print("No checkpoint found at {}".format(checkpoint_dir))


def evaluate(sentence):
  attention_plot = np.zeros((max_len, max_len))

  sentence = preprocess_sentence(sentence)

  inputs = [word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_len,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden , topic_vector = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([word_index['<start>']], 0)

  for t in range(max_len):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out , topic_vector)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += index_word[predicted_id] + ' '

    if index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()

def response(sentence):
  result, sentence, attention_plot = evaluate(sentence)
  x =0

  # print('Input: %s' % (sentence))
      
  if debug == True:
    print('Response: {}'.format(result)
  return result

  # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

batch_size = 1
Ty = 20
# def beam_search_decoder(sentence , beam_width = 3):
#   sentence = preprocess_sentence(sentence)
#   inputs = [word_index[i] for i in sentence.split(' ')]
#   inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                          maxlen=max_len,
#                                                          padding='post')
#   inputs = tf.convert_to_tensor(inputs)

#   predictions = [([],0)] * beam_width

#   # predictions = tf.convert_to_tensor(predictions)

#   enc_hidden = tf.zeros((1,units))

#   encoder_output , encoder_hidden , topic_vector = encoder(inputs , enc_hidden)

#   decoder_hidden = [encoder_hidden] * 3 


#   # decoder_hidden = [decoder_hidden] * beam_width

#   decoder_input = [tf.expand_dims([word_index['<start>'] ], 0)] * beam_width

#   # print("decoder_input" ,decoder_input[0])
#   print("decoder_hiddem:{} \n decoder_input: {} \n encoder_hidden :{} \n topic_vector {} \n".format(decoder_hidden[0].shape , decoder_input[0].shape , encoder_output.shape , topic_vector.shape))


#   for t in range(Ty):

#     current_predictions = []

#     for i in range(beam_width):

#       out , decoder_hidden[i] , _ = decoder(decoder_input[i] , decoder_hidden[i] , encoder_output , topic_vector)
#       # print("not done")
#       # print(out)

#       index = tf.argsort(out, axis=-1, direction='DESCENDING', stable=False, name=None)

#       prob = tf.sort(out ,axis = -1 , direction = 'DESCENDING')
#       # print("index: {} prob: {}".format((index[0]) , prob[0]));
#       # print("done")
#       # print(int(index[0][0]))
#       # print(int(index[0][1]))
#       # print(int(index[0][2]))
#       # print("done" ,current_predictions)
#       for j in range(beam_width):
#         if t ==0 and i ==1 :
#           continue
#           # print("do nothing!")
#         elif t == 0 and i ==2:
#           continue
#           # print("do nothing!")
#         else :
#           current_predictions.append((predictions[i][0] + [int(index[0][j])] , np.log(prob[0][j]) + predictions[i][1]))
#           # if t == 0 and i ==0 :
#             # print(current_predictions)
#             # print(count)
      
#       # print(current_predictions)

#     def get_prob(pred):
#       # print(pred[1])
#       return pred[1]

#     # print(current_predictions)
#     current_predictions = sorted(current_predictions , key = get_prob , reverse =True)

#     # print(current_predictions)

#     predictions = current_predictions[:beam_width]
#     # print(predictions)

#     current_predictions = [pred[0] for pred in current_predictions]

#     print(current_predictions[0])
#     # print(current_predictions[1])

#     decoder_input = [tf.expand_dims([tf.convert_to_tensor(pred[t])],0) for pred in current_predictions]

#     print(decoder_input[0])

#   output = []

#   for pred , prob in predictions:
#     out = []
#     s = ""
#     for p  in pred:
#       if index_word[p] != "<end>":
#         s += " " + index_word[p]
#       else :
#         break
#     output.append(s)

#   prob = [pred[1] / len(output[i])  for i , pred in enumerate(predictions)]

#   # for i in range(beam_width):
#   print("resposne :{} \n {} \n {} ".format(output[0] , output[1] ,output[2]))

#   return output , prob

def beam_search_decoder(sentence , beam_width = 3):
  sentence = preprocess_sentence(sentence)
  inputs = [word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_len,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  predictions = []

  # predictions = tf.convert_to_tensor(predictions)

  enc_hidden = tf.zeros((1,128))

  encoder_output , encoder_hidden , topic_vector = encoder(inputs , enc_hidden)

  decoder_hidden = encoder_hidden 


  # decoder_hidden = [decoder_hidden] * beam_width

  decoder_input = tf.expand_dims([word_index['<start>'] ], 0) 

  # print("decoder_input" ,decoder_input[0])
  # print("decoder_hiddem:{} \n decoder_input: {} \n encoder_output :{} \n topic_vector {} \n".format(decoder_hidden.shape , decoder_input.shape , encoder_output.shape , topic_vector.shape))
  

  out , hidden, _ = decoder(decoder_input , decoder_hidden , encoder_output , topic_vector)
 
  index = tf.argsort(out, axis=-1, direction='DESCENDING', stable=False, name=None)

  prob = tf.sort(out ,axis = -1 , direction = 'DESCENDING')

  terminal_sentences, decoder_hidden, predictions = [], [], []

  decoder_hidden = [hidden] * 3

  # print(decoder_hidden)




  for i in range(beam_width):
        predictions.append(([int(index[0][i])], np.log(prob[0][i])))
      
  # print(predictions[0][0])
  
  decoder_input = [tf.expand_dims(tf.convert_to_tensor(pred[0]),0) for pred in predictions]
  # print(decoder_input[0])
      

  for t in range(1,Ty):
        
        current_predictions = []
        for i in range(beam_width):
            out , decoder_hidden[i] , _ = decoder(decoder_input[i] , decoder_hidden[i] , encoder_output , topic_vector)
            # print("once")

            index = tf.argsort(out, axis=-1, direction='DESCENDING', stable=False, name=None)

            prob = tf.sort(out ,axis = -1 , direction = 'DESCENDING')

            for j in range(beam_width):
                current_predictions.append((predictions[i][0] + [int(index[0][j])] , np.log(prob[0][j]) + predictions[i][1] , i))
            
        def get_prob(pred):
          return pred[1]

        current_predictions = sorted(current_predictions , key = get_prob , reverse =True)

        current_predictions = current_predictions[:beam_width]

        # print("time_step {} {}".format(t ,current_predictions))

        hidden = []
        inputs = []
        pred = []

        for j in range(beam_width):
              if index_word[current_predictions[j][0][t]] == "<end>":
                    beam_width -= 1
                    terminal_sentences.append((current_predictions[j][0] , current_predictions[j][1]))
              else :
                    hidden.append(decoder_hidden[current_predictions[j][2]])
                    inputs.append(tf.expand_dims([tf.convert_to_tensor(current_predictions[j][0][t])],0)  )
                    pred.append((current_predictions[j][0] , current_predictions[j][1]))
        
        decoder_hidden = hidden
        decoder_input = inputs
        predictions = pred

        # print(decoder_input)

        if beam_width <= 0 :
              break

  for x in range(len(predictions)):
        terminal_sentences.append((predictions[x][0],predictions[x][1]))

  terminal_sentences = sorted(terminal_sentences , key = get_prob , reverse =True)

  output = []

  for pred , prob in terminal_sentences:
    out = []
    s = ""
    for p  in pred:
      if index_word[p] != "<end>":
        s += " " + index_word[p]
      else :
        break
    output.append(s)

  prob = [pred[1] / len(output[i])  for i , pred in enumerate(terminal_sentences)]

  # for i in range(beam_width):
  if debug == True:    
    print("resposne :{}  {} \n {} {} \n {} {} ".format(output[0] , prob[0] , output[1] ,prob[1] ,output[2] , prob[2]))

  return output , prob

def string_to_audio(input_string, delete):
    language = 'en'
    gen_audio = gTTS(text = input_string, lang=language, slow=False)
    gen_audio.save("Output.mp3")
    os.system("mpg123 Output.mp3")
    if (delete == True):
      os.remove("Output.mp3")


def get_transcript():
    mic = sr.Microphone()
    r = sr.Recognizer()
    print("Speak Now")
    with mic as source: 
        audio = r.listen(source, timeout=5, phrase_time_limit=10) 
        try :
            result = r.recognize_google(audio)
            print(result)
        except :
            return None
    return result


if __name__ == '__main__':

      parser = argparse.ArgumentParser(description='Conversational Bot')
      parser.add_argument('-d', '--debug', type = bool, default= False, help='set debug value')
      parser.add_argument('-o', '--options',type = str, help='set input option')

      args = parser.parse_args()
      # debug = True
      debug = args.debug

      # r = sr.Recognizer()
      # with sr.AudioFile( 'examples/voice_2.wav') as source:
    
        # audio_text = r.listen(source)
        # using google speech recognition
        
        # text = r.recognize_google(audio_text)
        # if debug == True:
              
          # print('Converting audio transcripts into text ...')
          # print(text)
      while True:
        result = get_transcript()

        out = response(result)
        
        # beam_search_decoder(result)

        string_to_audio(out , True)


