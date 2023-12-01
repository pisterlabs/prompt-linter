import argparse
import json
import os
import pickle
import re
import shutil
import string
from collections import OrderedDict
from html.parser import HTMLParser
from multiprocessing import Pool
from zipfile import ZipFile

import enchant
import gensim
import nltk
import numpy as np
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LsiModel, Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm

CHUNK_SIZE = 1000


en_us = enchant.Dict("en_US")
en_uk = enchant.Dict("en_UK")

nlp = spacy.load("en_core_web_lg")
stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
nlp.max_length = 50000000
tokenizer = RegexpTokenizer('\w+')


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    text = s.get_data()
    text = re.sub('[^a-z]quot[^a-z]', '"', text)
    text = re.sub('[^a-z]apos[^a-z]', "'", text)
    text = re.sub('[^a-z]amp[^a-z]', '&', text)
    text = re.sub('[^a-z]lt[^a-z]', '<', text)
    text = re.sub('[^a-z]gt[^a-z]', '>', text)

    return text


def get_wordnet_tag(tag):
    morphy_tag = {'NN': wordnet.NOUN, 'JJ': wordnet.ADJ,
                  'VB': wordnet.VERB, 'RB': wordnet.ADV}
    if tag[:2] not in morphy_tag:
        return None

    return morphy_tag[tag[:2]]


def do_preprocessing(token):
    text = token.lemma_

    # stopword removal
    if text in stop_words:
        return ''

    # check that it only has letters
    for c in text:
        if c not in string.ascii_letters:
            return ''

    return text


def process_text(text, preprocess):
    processed_words = []

    if isinstance(text, str):
        text = nlp(text)

    for token in text:
        if token.text in string.punctuation:
            continue

        if token.pos_ not in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            continue

        if preprocess:
            token = do_preprocessing(token)
        else:
            token = token.text
        if len(token) == 0:
            continue

        # Vocabulary check
#         if token not in vocab:
#             continue
        if not check_if_english_word(token):
            continue

        processed_words.append(token)
    return processed_words


def build_lsa(texts, size=300):
    dictionary = corpora.Dictionary(texts)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]

    model = LsiModel(doc_term_matrix, num_topics=300,
                     id2word=dictionary)  # train model

    return model, dictionary


def get_lsa_word_vectors(texts, lsa_model, dictionary):
    embeddings = {}
    for text in texts:
        for word in text:
            if word in embeddings:
                continue

            vec_bow = dictionary.doc2bow([word])
            vec_lsi = np.array([e[1] for e in lsa_model[vec_bow]])
            if len(vec_lsi) < lsa_model.num_topics:
                vec_lsi = np.hstack(
                    (vec_lsi, np.zeros(lsa_model.num_topics - len(vec_lsi))))
            embeddings[word] = vec_lsi
    return embeddings


def read_archive(archive_name, preprocess=False, split_sentences=False):
    texts = []

    with ZipFile(archive_name) as arch:
        for fname in arch.namelist():
            with arch.open(fname, 'r') as fin:
                text = fin.read().decode()

                text = strip_tags(text).lower()

                if split_sentences:
                    for sentence in nlp(text).sents:
                        texts.append(process_text(sentence, preprocess))
                else:
                    texts.append(process_text(text, preprocess))
    return texts


def process_single_text(text, preprocess, split_sentences):
    res = []
    if split_sentences:
        for sentence in nlp(text).sents:
            processed_sentence = process_text(sentence, preprocess)
            if len(processed_sentence) > 0:
                res.append(processed_sentence)
    else:
        res = [process_text(text, preprocess)]

    return res


def process_file(fname, preprocess=True, split_sentences=True):
    with open(fname, 'rt') as fin:
        text = fin.read()
        text = strip_tags(text).lower()

        return process_single_text(text, preprocess, split_sentences)


def process_magazine_file(fname, preprocess=True, split_sentences=True, min_text_length=10):
    split_sections = re.compile(r'##\d+\s+[Section]*\s*\:*')

    with open(fname, 'rt') as fin:
        text = fin.read().strip()
        texts = split_sections.split(text)

        res = []
        for text in texts:
            text = strip_tags(text).lower()
            if len(text) < min_text_length:
                continue
            for processed_block in process_single_text(text, preprocess, split_sentences):
                yield processed_block


def check_if_english_word(word):
    return en_us.check(word) or en_uk.check(word)


def load_quantiles_file(quantiles_file):
    with open(quantiles_file, 'rt') as fin:
        quantiles = json.load(fin)

    for arr in quantiles:
        for i in range(len(arr)):
            arr[i] = arr[i].replace(
                "/Users/rbotarleanu/Machine Learning/datasets/nlp/en",
                "/opt/rbotarleanu/datasets"
            )

    return quantiles


def load_files(quantiles_file):
    quantiles = load_quantiles_file(quantiles_file)
    files = OrderedDict()
    for i in range(len(quantiles)):
        qf = quantiles[i]
        if i > 0:
            qf = list(set(qf) - set(quantiles[i - 1]))
        files[f"Level {i + 1}"] = qf

    return files


def get_tf(texts):
    hist = {}
    for text in texts:
        for word in text:
            if word not in hist:
                hist[word] = 0
            hist[word] += 1
    return hist


def train_models(files, output_dir, vector_size=300, window=5, min_count=5,
                 epochs=5, workers=4):
    texts = []
    tf = []
    for file in files:
        print(file)
        print('\tPreprocessing', len(files[file]), 'files.')

        pool = Pool(workers)
        resp = pool.map_async(process_file, files[file], chunksize=CHUNK_SIZE)
        resp.wait()

        for r in resp.get():
            if len(r) > 0:
                texts += r

        tf.append(get_tf(texts))
        with open(f'{output_dir}/term_frequency.json', 'wt') as fout:
            json.dump(tf, fout)


        print('\tTraining Word2Vec on', len(texts), "texts.")
        model = gensim.models.Word2Vec(
            texts,
            size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers)
        model.train(texts, total_examples=len(texts), epochs=epochs)
        model.save(f'{output_dir}/{file}.model')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--min_count', '-m', type=int, default=5)
    parser.add_argument('--vector_size', '-s', type=int, default=300)
    parser.add_argument('--input_file', '-i', type=str,
                        default='coca_tasa_quantiles_cds_unsorted.json')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='quantile_models_cds_unsorted')
    parser.add_argument('--window_size', '-ws', type=int, default=5)
    parser.add_argument('--workers', '-w', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output_dir

    files = load_files(args.input_file)

    if os.path.exists(output_dir):
        print('Output directory already exists! Will delete...')
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    train_models(files, output_dir, window=args.window_size,
                 epochs=args.epochs, vector_size=args.vector_size,
                 workers=args.workers, min_count=args.min_count)
