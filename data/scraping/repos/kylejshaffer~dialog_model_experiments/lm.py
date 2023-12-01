import json
import math
import os
import sys

import keras.backend as K
import numpy as np
import tensorflow_datasets as tfds

from keras.models import Model
from keras.layers import Layer
from keras.layers import Input, Dense, Dropout, Embedding, GRU, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from pytorch_pretrained_bert import OpenAIGPTTokenizer

"""
IDEAS
1. Train on entire sequence of conversation to use LM score over entire convo.
2. Train on input-output pairs of conversation to use LM scores over input and generated output

"""

def load_cakechat_data_with_tok(data_path, tokenizer, max_len):
    tok_lines = []
    start_id = tokenizer.vocab_size # tokenizer.special_tokens['_start_']
    end_id = tokenizer.vocab_size + 1 # tokenizer.special_tokens['_delimiter_']
    with open(data_path, mode='r') as infile:
        for ix, line in enumerate(infile):
            sys.stdout.write('\r Loading line {}...'.format(ix))
            json_line = json.loads(line.strip())
            for utt in json_line:
                text = utt['text'].strip()
                # toks = tokenizer.tokenize(text)
                # tok_ids = tokenizer.convert_tokens_to_ids(toks)
                tok_ids = tokenizer.encode(text)
                if len(tok_ids) > max_len:
                    tok_ids = tok_ids[:max_len]
                tok_ids.append(end_id)
                tok_ids.insert(0, start_id)
                tok_lines.append(tok_ids)

    return tok_lines

def load_polar_data(data_path, tokenizer, max_len):
    tok_lines = []
    start_id = tokenizer.vocab_size
    end_id = tokenizer.vocab_size + 1
    with open(data_path, mode='r') as infile:
        for ix, line in enumerate(infile):
            sys.stdout.write('\rProcessing line {}...'.format(ix))
            l, _ = line.strip().split('\t')
            tok_ids = tokenizer.encode(l.strip())
            if len(tok_ids) > max_len:
                tok_ids = tok_ids[:max_len]
            tok_ids.append(end_id)
            tok_ids.insert(0, start_id)
            tok_lines.append(tok_ids)

    return tok_lines

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.015
    drop = 0.6
    epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1+epoch) / epochs_drop))
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# class BahdanauAttention(Model):
class BahdanauAttention(K.tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1) # originally `tf.expand_dims`

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(K.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = K.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class LanguageModel(object):
    def __init__(self, vocab_size, batch_size):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = 300
        self.hidden_dim = 1024
        self.hidden_dense_dim = 400
        self.lr_schedule = True
        self.rec_cell = LSTM
        self._build_model()

    def _build_model(self):
        in_words = Input(shape=(None,))
        embeddings = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)(in_words)
        embeddings = Dropout(0.3)(embeddings)
        encoded_1 = self.rec_cell(units=self.hidden_dim, return_sequences=True)(embeddings)
        encoded_2 = self.rec_cell(units=self.hidden_dim, return_sequences=True)(encoded_1)
        dense_hidden = Dense(units=self.hidden_dense_dim, activation='relu')(encoded_2)
        dense_hidden = Dropout(0.2)(dense_hidden)
        logits = Dense(units=self.vocab_size, activation='linear')(dense_hidden)

        lm = Model(inputs=in_words, outputs=logits)
        self.model = lm

    def _loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

    def train(self, train_data, valid_data, n_epochs):
        np.random.seed(7)

        x_train, y_train = train_data
        ckpt_fname = '/data/users/kyle.shaffer/chat_models/movie_lm_{epoch:02d}_{val_loss:.2f}.h5'
        ckpt = ModelCheckpoint(ckpt_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.summary()

        if self.lr_schedule:
            print('Setting up step-based LR schedule...')
            opt = Adagrad()
            lr_schedule = LearningRateScheduler(step_decay, verbose=1)
            self.model.compile(optimizer=opt, loss=self._loss)
            self.model.fit(x_train, y_train, validation_data=valid_data, batch_size=self.batch_size,
                       epochs=n_epochs, callbacks=[ckpt, lr_schedule])
        else:
            opt = 'adam'
            self.model.compile(optimizer=opt, loss=self._loss)
            self.model.fit(x_train, y_train, validation_data=valid_data, batch_size=self.batch_size,
                       epochs=n_epochs, callbacks=[ckpt])

if __name__ == '__main__':
    batch_size = 128
    n_epochs = 60
    max_len = 45

    train_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/corpora_processed/train_no_tok.txt'
    valid_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/corpora_processed/valid_no_tok.txt'
    vocab_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/tokens_index/t_idx_processed_dialogs.json'
    conditions_path = '/data/users/kyle.shaffer/dialog_data/cornell_movie/cakechat_model/conditions_index/conditions_index.json'
    train_polar_path = '/data/users/kyle.shaffer/dialog_data/polar/polar_train.txt'
    valid_polar_path = '/data/users/kyle.shaffer/dialog_data/polar/polar_valid.txt'
    
    bpe_tok_path = '/data/users/kyle.shaffer/dialog_data/movie_bpe.tok'

    # model_name = 'openai-gpt'
    # special_tokens = ['_start_', '_delimiter_', '_classify_']
    # gpt_tok = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    # print('GPT tokenizer initialized...')

    with open(vocab_path, mode='r') as infile:
        token_to_index = json.load(infile)
        token_to_index['<UNK>'] = max(token_to_index.values()) + 1
        index_to_token = {int(v): k for k, v in token_to_index.items()}

    # with open(conditions_path, mode='r') as infile:
    #     index_to_condition = json.load(infile)
    #     index_to_condition = {int(k): v for k, v in index_to_condition.items()}
    #     print(index_to_condition)
    
    # condition_to_index = {v: k for k, v in index_to_condition.items()}

    gpt_tok = tfds.features.text.SubwordTextEncoder.load_from_file(bpe_tok_path)
    # print(len(token_to_index))
    print('BPE vocab size:', gpt_tok.vocab_size + 2)
    train_vocab_size = gpt_tok.vocab_size + 2

    # Hard-coding vocab-size for now
    language_model = LanguageModel(vocab_size=train_vocab_size, batch_size=batch_size)

    # train_lines = load_cakechat_data(train_path, token_to_index, max_len)
    # valid_lines = load_cakechat_data(valid_path, token_to_index, max_len)
    train_lines = load_cakechat_data_with_tok(train_path, gpt_tok, max_len)
    valid_lines = load_cakechat_data_with_tok(valid_path, gpt_tok, max_len)
    print(len(train_lines), len(valid_lines))

    print()
    print('RAW TRAINING LINES')
    print(train_lines[0])
    print(train_lines[-1])
    print(valid_lines[0])
    print(valid_lines[-1])
    print()

    # train_polar_lines = load_polar_data(train_polar_path, bpe_tok, max_len)
    # valid_polar_lines = load_polar_data(valid_polar_path, bpe_tok, max_len)
    # print(len(train_polar_lines), len(valid_polar_lines))

    # train_lines = train_movie_lines + train_polar_lines
    # valid_lines = valid_movie_lines + valid_polar_lines
    # print(len(train_lines), len(valid_lines))

    x_train_lines = [i[:-1] for i in train_lines]
    y_train_lines = [i[1:] for i in train_lines]
    del train_lines

    x_valid_lines = [i[:-1] for i in valid_lines]
    y_valid_lines = [i[1:] for i in valid_lines]
    del valid_lines

    for i in range(10):
        print(x_train_lines[i])
    print()

    for i in range(10):
        print(y_train_lines[i])
    print()

    for i in range(10):
        print(x_valid_lines[i])
    print()

    for i in range(10):
        print(y_valid_lines[i])
    print()

    x_train, y_train = pad_sequences(x_train_lines, padding='post', maxlen=max_len), pad_sequences(y_train_lines, padding='post', maxlen=max_len)
    x_valid, y_valid = pad_sequences(x_valid_lines, padding='post', maxlen=max_len), pad_sequences(y_valid_lines, padding='post', maxlen=max_len)
    y_train = np.expand_dims(y_train, axis=-1)
    y_valid = np.expand_dims(y_valid, axis=-1)

    print()
    print('TRAINING MATRICES')
    print(x_train[0])
    print(y_train[0])
    print(x_valid[0])
    print(y_valid[0])
    print()

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_train.mean(), y_train.mean())
    print(x_valid.mean(), y_valid.mean())
    print(x_train.min(), x_valid.min())
    print(y_train.min(), y_valid.min())
    language_model.train([x_train, y_train], [x_valid, y_valid], n_epochs=n_epochs)
