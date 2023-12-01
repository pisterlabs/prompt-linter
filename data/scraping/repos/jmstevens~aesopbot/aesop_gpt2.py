import math
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import json
import os
import time
from ftfy import fix_text
#:os.chdir('../')
import pickle
import numpy as np
import string, os
from gensim.models import KeyedVectors
import gensim.downloader as api
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import random
import sys
from datetime import date
from collections import Counter
import matplotlib.pyplot as plt
from src.features.build import Lyrics
from src.features.transform_data import Transform
from random import shuffle
from tensorflow.python.framework import tensor_shape
from tokenizers import CharBPETokenizer, BertWordPieceTokenizer
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def load_data():
    data_dir = 'data/processed/verses.txt'
    with open(data_dir, "r") as fp:   # Unpickling
        lyrics = fp.read()
    lyrics_clean = clean_text(lyrics)


def word_based():
    _t = Lyrics(32,10000)
    #arr = _t.verse_lines
    corpus = _t.lyrics
    tokenizer = Tokenizer()
    def get_sequence_of_tokens(corpus):
        ## tokenization
        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1
        # convert data to sequence of tokens
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words
    inp_sequences, total_words = get_sequence_of_tokens(corpus)
    num_words = total_words
    print(inp_sequences[:10])

    input_sequences = inp_sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    #input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len+1, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    return tokenizer, num_words, tf.data.from_tensor_slices((predictors, label))



# In[ ]:


# def tf_encode(pt, en):
#     result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
#     result_pt.set_shape([None])
#     result_en.set_shape([None])
#
#     return result_pt, result_en
#
#
# def filter_max_length(x, y, max_length=MAX_LENGTH):
#     return tf.logical_and(tf.size(x) <= max_length,
#                         tf.size(y) <= max_length)
#
# def fetch_dataset(train_dataset, val_dataset, batch_size, padded_shapes=([-1], [-1]), epoch=25, buffer_size=10000):
#     train_dataset = train_dataset.map(tf_encode)
#     train_dataset = train_dataset.filter(filter_max_length)
#     # cache the dataset to memory to get a speedup while reading from it.
#     train_dataset = train_dataset.cache()
#     train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size)
#     train_dataset = train_dataset.repeat(epoch)
#     train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
#
#     val_dataset = val_dataset.map(tf_encode)
#     val_dataset = val_dataset.filter(filter_max_length).padded_batch(batch_size)
#     return train_dataset, val_dataset


def verse_pairs_approach(target_vocab_size=2**12):
    _t = Transform()
    arr = [i for i in _t.verse_lines if len(i) > 0]
    dataset = list()
    for verse in arr:
        if max([len(i.split()) for i in verse]) > 1 and max([len(i.split()) for i in verse]) < 25:
            chunk_number = len(verse) // 4
            # chunks = [verse[x:x+chunk_number] for x in range(0, len(verse), chunk_number)]
            if chunk_number != 0:
                chunks = ['<START> ' + ''.join([ j + ' <NEWLINE> ' for j in verse[x:x+chunk_number]]) + ' <END>' for x in range(0, len(verse), chunk_number)]
                chunks = [chunk for chunk in chunks if len(chunk.split('<NEWLINE>')) > 2]
                dataset.append((chunks[:2], chunks[2:]))
    # for i in arr:
    #     tmp = [ ' <NEWLINE>  '.join([clean_text(j[0]), clean_text(j[1])]) for j in zip(i[0::2],i[1::2])]
    #     dataset.append([z for z in zip(tmp[0::2], tmp[1::2])])
    example = [x[0] for x in dataset]
    target = [x[1] for x in dataset]
    print(example[:2], target[:2])
    X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)
    len(X_train)
    train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size, reserved_tokens=['<UNK>','<NEWLINE>','<START>','<END>'])#, reserved_tokens=['<UNK>'])

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size,reserved_tokens=['<UNK>','<NEWLINE>','<START>','<END>']) #reserved_tokens=['<UNK>'])

    BUFFER_SIZE = 15000
    BATCH_SIZE = 32

    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    MAX_LENGTH = 125



    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)


    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

    return train_dataset, val_dataset, tokenizer_en, tokenizer_pt



def verse_by_verse(test_size=.10, shuffle=False, target_vocab_size=2**12):
    _t = Transform()
    arr = _t.verse_lines
    dataset = list()
    for verse in arr:
        x = verse[0::2]
        y = verse[1::2]
        #[print(i) for i in zip(x, y)]
#         dataset +=
    #print(dataset[0])
    if shuffle:
        np.random.shuffle(dataset)
    train = dataset[:round(len(dataset) * test_size)]
    test = dataset[round(len(dataset) * test_size):]

    train_examples = tf.data.Dataset.from_tensor_slices(train)
    val_examples = tf.data.Dataset.from_tensor_slices(test)
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)
    return train_examples, val_examples, tokenizer_en, tokenizer_pt


def fill_in_the_blank(test_size=.10, shuffle=False, target_vocab_size=2**12):
    _t = Transform()
    arr = _t.verse_lines
    data_dir = 'data/processed/verses.txt'
    with open(data_dir, "rb") as fp:   # Unpickling
        lyrics = pickle.load(fp)
    arr = [[j for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j] for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]#tokenizer = BertWordPieceTokenizer()
    #tokenizer.train(['data/processed/verses_encoded.txt'])
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #special_tokens_dict = {'bos_token':'|START|', 'eos_token':'|END|', 'unk_token':'|UNK|', 'sep_token':'|SEP|', 'pad_token':'|PAD|', 'cls_token':'|CLS|', 'mask_token':'|MASK|'}
    #num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #print('We have added', num_added_toks, 'tokens')
    #model.resize_token_embeddings(len(tokenizer))
    #tokenizer.add_tokens(['<START>','<END>'])
    dataset = list()
    for verse in arr:
        num_times = random.randint(1, 5)
        try:
            if max([len(i.split()) for i in verse]) > 1 and max([len(i.split()) for i in verse]) < 50:
                chunk_number = len(verse) // 3
                chunks = [verse[x:x+chunk_number] for x in range(0, len(verse), chunk_number)]
                #chunks = ['<START> ' + ''.join([ j for j in verse[x:x+chunk_number]]) for x in range(0, len(verse), chunk_number)]
                #chunks = [chunk for chunk in chunks if len(chunk.split('<NEWLINE>')) > 2]
                chunk_list = [' '.join(chunk_verse).split() for chunk_verse in chunks]

                for chunk in chunk_list:
                    for _ in range(0, num_times,1):
                        mask = np.random.random(len(chunk))
                        mask_bool = random.uniform(.3, .4)
                        mask_x = mask > mask_bool
                        mask_y = mask < mask_bool
                        x = '<START> ' + ' '.join(['[MASK]' if not x else chunk[i] for i, x in enumerate(mask_x)]) + ' <END>'
                        #x = ' '.join(np.array(verse)[mask_x].tolist())
                        #y = ' '.join(np.array(chunk).tolist())
                        #$y = ' '.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
                        #y = '|<GAP>|'.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
                        y = '<START> ' + ' '.join(['[MASK]' if x else chunk[i] for i, x in enumerate(mask_x)]) + ' <END>'
                         # = ' '.join([np.array(i)[mask_y] for i in chunk])
                        # x = ' '.join(np.array(chunk)[mask_x].tolist())
                        # y = ' '.join(np.array(chunk)[mask_y].tolist())
                        #x = ' '.join([' ' if not x else chunk.split(' ')[i] for i, x in enumerate(mask_x)])
                        #x = ' '.join([' ' if not x else chunk.split(' ')[i] for i, x in enumerate(mask_x)])
                        #y = chunk
                        dataset.append((x, y))
        except ValueError:
            pass
    print(dataset[0])
    example = np.array(pad_sequences([tokenizer.encode(x[0]) for x in dataset], padding='post'))
    target = np.array(pad_sequences([tokenizer.encode(x[1]) for x in dataset], padding='post'))

#    target = [tokenizer.encode(x[1]).ids for x in dataset]

    print(len(dataset))
    print(dataset[0])
    X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)
    train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    #tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #    (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)#, reserved_tokens=['<UNK>'])

    #tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #    (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)#,reserved_tokens=['<UNK>'])

    BUFFER_SIZE = 15000
    BATCH_SIZE = 64

    def encode(lang1, lang2):
        lang1 = [tokenizer.get_vocab_size()] + tokenizer.encode(lang1.numpy()).ids + [tokenizer.get_vocab_size()+1]
        lang2 = [tokenizer.get_vocab_size()] + tokenizer.encode(lang2.numpy()).ids + [tokenizer.get_vocab_size()+1]
        return lang1, lang2
#
    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
#
        return result_pt, result_en
#
    MAX_LENGTH = 125
#
#
#
    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)


    #train_dataset = train_examples.map(tf_encode)
    #train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_examples.cache()
#    train_dataset = train_dataset.repeat(25)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    #val_dataset = val_examples.map(tf_encode)
    val_dataset = val_examples.padded_batch(BATCH_SIZE)

    return train_dataset, val_dataset, tokenizer#, tokenizer_pt


def window_based(test_size=.10, shuffle=False, target_vocab_size=2**12):
    test_size = 1 - test_size
    dataset = list()
    _t = Lyrics(32, 1000)
    data_dir = 'data/processed/verses_encoded.txt'
    with open(data_dir, "r") as fp:   # Unpickling
        lyrics = fp.read()
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(['data/processed/verses_encoded.txt'])
    tokenizer.add_tokens(['<START>','<END>','<NEWLINE>'])
    arr = [[clean_text(j).replace('newline','<NEWLINE>').replace('start','<START>').replace('end','<END>') for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j] for i in list(np.array(lyrics.split('\n\n'))) if len(i.split(' \n ')) > 0]

    # print(arr)
    # for verse in arr:
    #     chunk_number = len(verse) // 5
    #     if chunk_number > 0:
    #         chunks = ['<START> ' + ''.join([ j.replace('\n','').replace('\n\n','') + ' <NEWLINE> ' for j in verse[x:x+chunk_number]]) + ' <END>' for x in range(0, len(verse), chunk_number)]
    #         chunks = [chunk for chunk in chunks if len(chunk.split('<NEWLINE>')) > 2]
    #         print()
    #         dataset.append(chunks)
    # train = dataset
    train = [y for x in arr for y in x]
    train = [tokenizer.encode(i).ids for i in train]
    train = [y for x in train for y in x]
    # train.split('<NEWLINE>')
    # print(train)
    # train = ' <EOV> '.join(dataset)
    # print(train)
    # tokenizer.add_tokens(['<START>','<END>','<NEWLINE>','<EOV>'])
    # target = _t.target
    # target = [x[1] for x in dataset]
    # print(len(dataset))
    # X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)

    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # print(len(dataset))
    # np.random.shuffle(dataset)
    # train_test = dataset[:round(len(dataset) * test_size)]
    # train = train_test[:round(len(train_test) * test_size)]
    # test = train_test[round(len(train_test) * test_size):]
    # val = dataset[round(len(dataset) * test_size):]
    # train_dataset = tf.data.Dataset.from_tensor_slices(train)

    # tokenizer = BertWordPieceTokenizer("data/processed/vocab.txt", lowercase=True)
    # tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for en in train_dataset), target_vocab_size=target_vocab_size, reserved_tokens=['<UNK>','<NEWLINE>','<START>','<END>'])
    train_dataset = tf.data.Dataset.from_tensor_slices(train)
    seq_length = 40
    # examples_per_epoch = len(train.split())//(seq_length+1)

    # data = [i for i in flattened_list if len(i) < 100]
    sequences = train_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 128

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 20000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    dataset = dataset.repeat(50)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(dataset)
    return dataset, tokenizer

def simple_method(sequence_size, testSetRatio=0.15):
    testSetRatio = 1-testSetRatio
    data_dir = 'data/processed/verses_test.txt'
    with open(data_dir, "rb") as fp:   # Unpickling
        lyrics = pickle.load(fp)
    arr = [' <NEWLINE> '.join([clean_text(j) for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j]) for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]
    #tokenizer = BertWordPieceTokenizer()
    #tokenizer.train(['data/processed/verses_encoded.txt'])
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    #tokenizer.train(['data/processed/verses_encoded.txt'])
    special_tokens_dict = {'eos_token':'<END>','sep_token':'<NEWLINE>','bos_token':'<START>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.encode(' <NEWLINE> '))
    tokenizer.save_pretrained('src/data/tokenizers')
    dataset = list()
    for verse in arr:
        tmp = list()
        verse = ' <START> ' + verse + ' <END> '
        verse_split = verse.split(' <NEWLINE> ')
        for line in verse_split:
            tmp = tmp + tokenizer.encode(line + ' <NEWLINE>', add_prefix_space=True)
        if tmp:
            dataset.append(tmp)
    print(dataset[0])
    # dataset = [[item for sublist in verse.split(' \n ') for tokenizer.encode(item, add_prefix_space=True) in sublist] for verse in arr]
    np.random.shuffle(dataset)
    verse_length = [len(verse) for verse in dataset]
    verse_average = sum(verse_length) / len(verse_length)

    print(f'Average number of words in a verse {verse_average}')
    # dataset = dataset[
    train = dataset[:round(len(dataset) * testSetRatio)]
    test = dataset[round(len(dataset) * testSetRatio):]
    print(f'train size {len(train)}')
    print(f'test size {len(test)}')
    trainTensor = simple_pipeline(train, sequence_size)
    testTensor = simple_pipeline(test, sequence_size)

    return trainTensor, testTensor, tokenizer

def simple_pipeline(dataset, sequence_size):
    dataset = [y for x in dataset for y in x]
    assert isinstance(dataset[0], int)
    print(f'number of tokens {len(dataset)}: \n{dataset[:5]}')
    train = tf.data.Dataset.from_tensor_slices(dataset)
    train = train.window(sequence_size, drop_remainder=True)
    for window in train.take(5):
        print(list(window.as_numpy_iterator()))

    train = train.flat_map(lambda window: window.batch(sequence_size))
    train = train.shuffle(10000).batch(64)
    train = train.map(lambda windows: (windows[:,:-1], windows[:,1:]))
    # train = train.cache()
    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    return train


def gelu(x):
    with tf.name_scope("gelu"):
        cdf = 0.35 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def shape_as_list_2(x):
    return [int(i) for i in tf.shape(x)]


def get_padding_mask(seq):
    with tf.name_scope("Padding_Mask"):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def attention_mask(size):
    with tf.name_scope("attention_mask"):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask # (seq_len, seq_len)

def create_masks(inp):
    with tf.name_scope("attn_masking"):
        # Encoder padding mask
        att_mask = attention_mask(tf.shape(inp)[1])

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        padding_mask = get_padding_mask(inp)
        mask = tf.maximum(padding_mask, att_mask)


    return mask





def scaled_dot_product_attention(q, k, v, training, mask=None):
    """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
          but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights


    """
    matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len, seq_len_k)

    # scale matmul_qk
    if self.scale:
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) for scores to add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None
    )
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att_dropout=0.4,
                 residual_dropout=0.45, scale=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.att_dropout = att_dropout
        self.residual_dropout=residual_dropout
        self.scale=scale
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.c_attn = Conv1d(self.d_model, self.d_model * 3)
        self.c_proj = Conv1d(self.d_model, self.d_model)

    def multihead_attention(self, q, k, v, training, mask=None):
        """
            Calculate the attention weights.
            q, k, v must have matching leading dimensions.
            k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
            The mask has different shapes depending on its type(padding or look ahead)
              but it must be broadcastable for addition.

          Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable
                  to (..., seq_len_q, seq_len_k). Defaults to None.

          Returns:
            output, attention_weights


        """
        matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len, seq_len_k)

        # scale matmul_qk
        if self.scale:
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            matmul_qk = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor
        if mask is not None:
            matmul_qk += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) for scores to add up to 1
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)

        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.att_dropout, name="attn_dropout")   # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
           Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def merge_heads(self, x):
        batch_size = tf.shape(x)[0]

        scaled_attention = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        merged = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        return merged

    def call(self, x, mask=None, past_layer=None, training=True):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)

        query = self.split_heads(query) # (batch_size, seq_len, d_model)
        key = self.split_heads(key) # (batch_size, seq_len, d_model)
        value = self.split_heads(value) # (batch_size, seq_len, d_model)

        if past_layer is not None:
            past_key, past_value = tf.unstack(past_layer, axis=1)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=2)

        present = tf.stack([key, value], axis=1)
        scaled_attention, attention_weights = self.multihead_attention(query, key, value, training, mask)

        concat_attention = self.merge_heads(scaled_attention)

        output = self.c_proj(concat_attention)

        if training:
            output = tf.nn.dropout(output, rate=self.residual_dropout, name="resit_dropout")

        return output, present


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.stddev = stddev
        self.mean = mean
        self.initializer = initializer
        if self.initializer is None:
            self.initializer = tf.random_normal_initializer(mean=self.mean, stddev=self.stddev)

    def build(self, input_shape):
        with tf.name_scope("embedding_weights"):
            self.embedding_weights = self.add_weight("weights", shape=[self.vocab_size, self.embedding_size],
                                                     dtype="float32",
                                                     initializer=self.initializer
                                                     )
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mode="embedding", scale=False):
        if mode == "embedding":
            return self.embedding(inputs, scale=scale)
        elif mode == "projection":
            return self.projection(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def embedding(self, inputs, scale=False):
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
            inputs = tf.cast(inputs, tf.int32)
            embeddings = tf.nn.embedding_lookup(self.embedding_weights, inputs)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            if scale:
                embeddings *= self.embedding_size ** 0.5

            return embeddings

    def projection(self, inputs):
        with tf.name_scope("output_layer"):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            h_flat = tf.reshape(inputs, [-1, self.embedding_size])
            logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])



class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, position_seq, pos_embedding_size, trainable=True, stddev=0.02, mean=0.0):
        super(PositionEmbeddingLayer, self).__init__()
        self.position_seq = position_seq
        self.hidden_size = pos_embedding_size
        self.trainable = trainable
        self.stddev = stddev
        self.mean = mean

        if trainable:
            self.position_embedding = EmbeddingLayer(self.position_seq, self.hidden_size,
                                                     stddev=self.stddev, mean=self.mean)

    def call(self, inputs, start=1):
        with tf.name_scope("pos_embedding"):
            if self.trainable:
                batch_size = tf.shape(inputs)[0]
                batch_seq = tf.shape(inputs)[1]

                positions = tf.reshape(tf.tile(tf.range(start, batch_seq + start), [batch_size]),
                                       [batch_size, batch_seq])

                positions = tf.cast(positions, tf.int32)
                position_mask = tf.cast(tf.not_equal(inputs, 0), tf.int32)
                positions *= position_mask

                return self.position_embedding(positions)
            else:
                return self.get_position_sinusoid(self.position_seq)

    @staticmethod
    def get_position_sinusoid(seq_len, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
        position = tf.cast(tf.range(seq_len), tf.float32)
        num_timescales = hidden_size // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal



class Conv1d(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 filter_size,
                 weights_init_stdev=0.02,
                 weights_mean=0.0,
                 bias_init=0.0):
        super(Conv1d, self).__init__()

        self.weights_init_stdev = weights_init_stdev
        self.weights_mean = weights_mean
        self.bias_init = bias_init
        self.hidden_size = hidden_size
        self.filter_size = filter_size

    def build(self, input_shape):
        self.weight = self.add_weight(
            "cov1d_weights",
            shape=[self.hidden_size, self.filter_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=self.weights_init_stdev,
                mean=self.weights_mean))

        self.bias = self.add_weight("conv1d_biases",
                                    shape=[self.filter_size],
                                    initializer=tf.constant_initializer(self.bias_init))
        super(Conv1d, self).build(input_shape)

    def call(self, inputs):
        output_shape = [tf.shape(inputs)[0], tf.shape(inputs)[1]] + [self.filter_size]
        inputs = tf.reshape(inputs, [-1, self.hidden_size])  # shape [batch, seq , features] => [batch*seq, features]
        outputs = tf.matmul(inputs, self.weight) + self.bias
        outputs = tf.reshape(outputs, output_shape)  # Reshape => [batch, seq, filter_size]
        return outputs

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, hidden_size, filter_size, dropout_rate=0.45, activation=tf.nn.relu):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.dense_layer = Conv1d(self.hidden_size, self.filter_size)
        self.output_dense_layer = Conv1d(self.filter_size, self.hidden_size)

    def call(self, x, training=False):
        output = self.dense_layer(x)
        output = self.activation(output)
        output = self.output_dense_layer(output)

        if training:
            output = tf.nn.dropout(output, rate=self.dropout_rate, name="feed_forward_dropout")

        return output


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.gamma = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.beta = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, epsilon=1e-6, input_dtype=tf.float32):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        normalized = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return tf.cast(normalized * self.gamma + self.beta, input_dtype)

def argmax(logits):
	return tf.argmax(logits)


def top_k_logits(logits, k):
	if k == 0:
		return logits

	values, _ = tf.nn.top_k(logits, k=k)
	min_values = values[:, -1]

	return tf.where(
		logits < min_values,
		tf.ones_like(logits, dtype=logits.dtype) * -1e10,
		logits
	)

	# Nucleas Sampling (https://arxiv.org/pdf/1904.09751.pdf)


def top_p_logits(logits, p):
	"""Took from OpenAI GPT-2 Implememtation"""
	batch = tf.shape(logits)[0]
	sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
	cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
	indices = tf.stack([
		tf.range(0, batch),
		tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
	], axis=-1)
	min_values = tf.gather_nd(sorted_logits, indices)
	return tf.where(
		logits < min_values,
		tf.ones_like(logits) * -1e10,
		logits,
	)

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets"),
    tf.TensorSpec(shape=(None), dtype=tf.int32, name="Step")
]


test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets"),
    tf.TensorSpec(shape=(None), dtype=tf.int32, name="Step")
]

class Gpt2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, vocab_size, tokenizer,
                 optimizer="adam", learning_rate=0.005, rev_embedding_projection=True):
        super(Gpt2, self).__init__()

        self.rev_embedding_projection = rev_embedding_projection
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.optimizer_t = optimizer
        self.dataset = None
        self.mirrored_strategy = None

        self.embedding = EmbeddingLayer(
            self.vocab_size, self.d_model)

        self.pos_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.d_model)

        self.decoder_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff)
                               for _ in range(self.num_layers)]
        self.layer_norm = LayerNormalization(self.d_model)

        if not self.rev_embedding_projection:
            self.output_layer = OutputLayer(self.vocab_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(
            name='accuracy')

        self.train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

        self.test_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

    def call(self, x, training=True, past=None):
        x = tf.cast(x, tf.int32)
        batch, sequence = tf.shape(x)[0], tf.shape(x)[1]
        if past is None:
            pasts = [None] * self.num_layers
        else:
            pasts = past

        assert len(pasts) == self.num_layers

        att_mask = create_masks(x)
        past_length = 1 if past is None else tf.shape(past)[-2]
        with tf.name_scope("embeddings"):
            embedded_x = self.embedding(x)
            hidden_states = embedded_x + self.pos_embedding(x, start=past_length)

        presents = []
        for decoder_layer, past in zip(self.decoder_layers, pasts):
            hidden_states, present = decoder_layer(hidden_states, training, att_mask, past=past)
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)

        if self.rev_embedding_projection:
            logits = self.embedding(hidden_states, mode="projection")
        else:
            logits = self.output_layer(hidden_states)

        return logits, presents

    @staticmethod
    def get_padded_accuracy(labels, logits):
        with tf.name_scope("padded_accuracy"):
            weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

            outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            padded_labels = tf.cast(labels, tf.int32)

            nonpad_seq = tf.math.count_nonzero(weights, dtype=tf.dtypes.float32, )
            acc = tf.cast(tf.equal(outputs, padded_labels), tf.float32)

            accuracy = tf.reduce_sum(tf.cast(acc * weights, tf.float32)) / nonpad_seq
            return tf.cast(accuracy, tf.float32)

    def create_optimizer(self):
        optimizer = self.optimizer_t.lower()
        with tf.name_scope("optimizer"):
            if optimizer == "adam":
                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                                                          epsilon=1e-9)
            elif optimizer == "adadelta":
                self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
            elif optimizer == "rms":
                self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            else:
                self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
            return self.optimizer

    def get_loss(self, real, pred):
        with tf.name_scope("loss_layer"):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.loss_object(real, pred)

            with tf.name_scope("loss_masking"):
                mask = tf.cast(mask, dtype=loss_.dtype)
                loss_ *= mask
            loss_ = tf.reduce_sum(loss_, axis=1)
            sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
            return sequence_avg_loss

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        with tf.name_scope('checkpoint_manager'):
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

            if load_model:  # If want to load trained weights
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored...............')
            else:
                print("Initializing model from scratch..........")

    def load_model(self, filepath):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, filepath, max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Model Restored..........................")

    def create_summary_writer(self, summary_path):
        train_summary_path = summary_path + "/train"
        test_summary_path = summary_path + "/test"

        with tf.name_scope('summary'):
            self.train_writer = tf.summary.create_file_writer(train_summary_path)
            self.test_writer = tf.summary.create_file_writer(test_summary_path)

            return self.train_writer, self.test_writer

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs, targets, step, grad_clip=True, clip_value=2.5):

        with tf.GradientTape() as tape:
            predictions, _ = self(inputs, training=True)
            loss = tf.reduce_mean(self.get_loss(targets, predictions))

        with tf.name_scope("gradients"):
            gradients = tape.gradient(loss, self.trainable_variables)
            if grad_clip:
                gradients = [(tf.clip_by_value(grad, -clip_value, clip_value))
                             for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        accuracy = self.get_padded_accuracy(targets, predictions)

        with tf.name_scope("summary_writer"):
            with self.train_writer.as_default():
                tf.summary.scalar("loss", loss, step=tf.cast(step, tf.int64))
                tf.summary.scalar("accuracy", accuracy, step=tf.cast(step, tf.int64))

        return loss, accuracy


    @tf.function(input_signature=test_step_signature)
    def test_step(self, inputs, targets, step, grad_clip=True, clip_value=2.5):

        with tf.GradientTape() as tape:
            predictions, _ = self(inputs, training=False)
            test_loss = tf.reduce_mean(self.get_loss(targets, predictions))
        test_accuracy = self.get_padded_accuracy(targets, predictions)

        with tf.name_scope("summary_writer"):
            with self.test_writer.as_default():
                tf.summary.scalar("test_loss", test_loss, step=tf.cast(step, tf.int64))
                tf.summary.scalar("test_accuracy", test_accuracy, step=tf.cast(step, tf.int64))

        return test_loss, test_accuracy

    def fit(self, train_dataset, test_dataset, EPOCHS=50):
        for epoch in range(EPOCHS):
            tf.summary.trace_on(graph=True, profiler=True)
            print('EPOCH :{}'.format(epoch))
            if not epoch == 0:
                step = epoch * step
                test_step = epoch * test_step
            tf.summary.trace_on(graph=True, profiler=True)
            for (step, (inputs, targets)) in enumerate(train_dataset):
                train_loss, train_acc = self.train_step(inputs, targets, step)
                if step % 100 == 0:
                    print('Step {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
                        step, train_loss, train_acc))

                if step == 25:
                    with self.train_writer.as_default():
                        tf.summary.trace_export(
                            name="gpt-2",
                            step=step,
                            profiler_outdir='logs/train')

                if step % 5000 == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print('Saving checkpoint for step {} at {}'.format(step,
                                                                       ckpt_save_path))
            # tf.summary.trace_on(graph=True, profiler=True)
            for (test_step, (inputs, targets)) in enumerate(test_dataset):
                test_loss, test_acc = self.test_step(inputs, targets, test_step)
                if not epoch == 0:
                    test_step = epoch * test_step
                if test_step % 100 == 0:
                    print('Step {} Test_Loss {:.4f} Test_Accuracy {:.4f}'.format(
                        test_step, test_loss, test_acc))

                if test_step == 25:
                    with self.test_writer.as_default():
                        tf.summary.trace_export(
                            name="gpt2_test",
                            step=test_step,
                            profiler_outdir='logs/test')

    def beam_search(self, predictions, top_k=25):
        #start with an empty sequence with zero score
        output_sequences = [([], 0)]

        #looping through all the predictions
        for token_probs in predictions:
            new_sequences = []

            #append new tokens to old sequences and re-score
            for old_seq, old_score in output_sequences:
                for char_index in range(len(token_probs)):
                    new_seq = old_seq + [char_index]
                    #considering log-likelihood for scoring
                    new_score = old_score + math.log(token_probs[char_index])
                    new_sequences.append((new_seq, new_score))

            #sort all new sequences in the de-creasing order of their score
            output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)

            #select top-k based on score
            # *Note- best sequence is with the highest score
            output_sequences = output_sequences[:top_k]

        return output_sequences

    def sample_sequence(self,seq_len, context=None,temperature=.96,
    						top_k=25,
    						top_p=.95,
				nucleus_sampling=True):
        # vocab_size=2**15
        # model_gen = Gpt2(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads, dff=self.dff, max_seq_len=self.max_seq_len, vocab_size=self.tokenizer.get_vocab_size(), tokenizer=self.tokenizer, optimizer="adam")
        # model_gen.create_optimizer()
        # model_gen.create_checkpoint_manager('checkpoint')
        bos=self.tokenizer.bos_token_id#.encode('<START>')#.ids[0]
        eos=self.tokenizer.eos_token_id#.ids[0]
        if context == None:
            print("Give some context to model.................")
            return
        context_str = context
        context = tf.expand_dims(([bos] + self.tokenizer.encode(context)), 0)
#        context = tf.expand_dims(([bos] + [self.tokenizer.encode(context)]), 0)
        prev = context
        print(prev)
        output = context
        past = None
        for i in range(seq_len):
            #context = tf.expand_dims((self.tokenicontext).ids), 0)
            #prev = context
            #output = context
            past = None

            logits, past = self(prev, training=False, past=past)
            # print(logits)
            #logits = (tf.nn.softmax(logits[-1, -5:, :].numpy(),axis=-1) / tf.cast(1.25, tf.float32)).numpy()
            logits = logits[:,-1,:] / tf.cast(temperature, tf.float32)
            #predictions = beam_search_decoder(logits, 5)
            #np.random.shuffle(predictions)
            #print([self.tokenizer.decode(i) for i in predictions])
            #predictions = predictions[0][0]
            # print(logits)
            logits = top_k_logits(logits, k=top_k)
            # print(logits)
            if nucleus_sampling:
            	logits = top_p_logits(logits, p=top_p)

            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)

            if tf.equal(samples, eos):
                print("Predicted end of sequence.")
                break

            # print("shape.........")
            # print(tf.shape(output))
            # print(tf.shape(samples))
            #context_str = context_str + ' ' + self.tokenizer.decode(predictions)
            #context = tf.expand_dims(([bos] + self.tokenizer.encode(context_str), 0))
            prev = samples
            output = tf.concat([output, samples], axis=-1)

            # print(tf.shape(output))
            # print(output)

        # print("--------------------------")
        result = tf.squeeze(output, axis=0)
        pred = [int(i) for i in result]
        generated_seq = self.tokenizer.decode([i for i in pred[1:]])
        #generated_seq = generated_seq.replace("|SEP|", "\n")
        generated_seq = ' '.join(generated_seq.split())
        generated_seq = generated_seq.replace("<NEWLINE>", "\n").replace("<|>","\n").replace("<|NEWLINE|NEWLINE|>","\n").replace("<|NEWLINE|NEWLINE|NEWLINE|>","\n")
        return generated_seq

from math import log
from numpy import array
from numpy import argmax

# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, proj_weights=None, kernel_initializer=None):
        super(OutputLayer, self).__init__()
        self.proj_weights = proj_weights
        self.output_dim = output_dim
        self.layer_weights = None
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        if self.proj_weights is None:
            input_dim = tensor_shape.dimension_value(input_shape[-1])
            self.layer_weights = self.add_weight(
                'output_layer_weights',
                shape=[input_dim, self.output_dim],
                initializer=self.kernel_initializer,
                trainable=True)
        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
        with tf.name_scope("residual_conn"):
            x = x + out
        out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        return x, present

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,
                 dr_rate=0.45):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dr_rate = dr_rate

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = FeedForward(self.d_model, self.dff, self.dr_rate)
        self.layer_norm1 = LayerNormalization(self.d_model)
        self.layer_norm2 = LayerNormalization(self.d_model)

    def call(self, x, training, mask, past=None):
        out, present = self.mha(self.layer_norm1(x), mask=mask, past_layer=past,
                                training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        return x, present


def run():
    sequence_size = 12
    trainTensor, testTensor, tokenizer = simple_method(sequence_size)
    model = Gpt2(6, 512, 8, 512, sequence_size, vocab_size=tokenizer.vocab_size+3, tokenizer=tokenizer, optimizer='adam')
    opt = model.create_optimizer()
    model.create_checkpoint_manager('checkpoint')
    model.create_summary_writer('logs')
    model.compile(loss=model.loss_object, optimizer=opt)
    model.fit(trainTensor, testTensor)
    # model.save('aesop')
