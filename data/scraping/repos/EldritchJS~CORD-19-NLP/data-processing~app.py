import csv
import os
import logging
import argparse
from os import environ
import json
from collections import defaultdict
from itertools import islice

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
spacy.prefer_gpu()
import spacy.cli
spacy.cli.download("en_core_web_sm")
import en_core_web_sm
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

import pyLDAvis.gensim

from kafka import KafkaProducer
from json import dumps

def main(args):
    nlp=en_core_web_sm.load()
    nlp.Defaults.stop_words.update(['sars','covid-19', 'cov-2','=','from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
    
    def lemmatize(doc):
        doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)
    
    def clean_tokens(doc):
        doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True and (len(token)<=4)!=True]
        return doc

    def generate_corpus(doc_list):
        words=corpora.Dictionary(doc_list)
        corpus = [words.doc2bow(doc) for doc in doc_list]
        return [words,corpus]

    def generate_lda_model(words,corpus):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=words,
                                                num_topics=10, 
                                                random_state=2,
                                                update_every=1,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)  
        return lda_model

    nlp.add_pipe(lemmatize,name='lemmatize',after='ner')
    nlp.add_pipe(clean_tokens, name="cleanup", last=True)

    fulltext_list = []
    abstract_list = []
    nlp.max_length = 2000000
    dataset_limit = 100

    with open(args.source) as f_in:
        reader = csv.DictReader(f_in)
        for row in islice(reader, 0, dataset_limit):
            abstract = row['abstract']
            nlp_abstract = nlp(abstract)
            abstract_list.append(nlp_abstract)

    abstract_gen_corp = generate_corpus(abstract_list)
    abstract_words=abstract_gen_corp[0]
    abstract_corpus=abstract_gen_corp[1]
    abstract_lda_model = generate_lda_model(abstract_words,abstract_corpus)
    logging.info('Topics: ')
    topics = []
    for topic in abstract_lda_model.show_topics():
        logging.info(topic[1])
        topics.append(topic[1])
    logging.info('creating kafka producer')
    data = {
            "topics": topics 
            }
    producer = KafkaProducer(bootstrap_servers=args.brokers,
                             value_serializer=lambda x: 
                             dumps(x).encode('utf-8'))
    producer.send(args.topic, value=data)

def get_arg(env, default):
    return os.getenv(env) if os.getenv(env, "") != "" else default

def parse_args(parser):
    args = parser.parse_args()
    args.source = get_arg('SOURCE_FILE', args.source)
    args.brokers = get_arg('KAFKA_BROKERS', args.brokers)
    args.topic = get_arg('KAFKA_TOPIC', args.topic)
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('parsing args')
    parser = argparse.ArgumentParser(description='process a CORD-19 file')
    parser.add_argument(
            '--source',
            help='Source file, env variable SOURCE_FILE',
            default='/mnt/data/metadata.csv')
    parser.add_argument(
            '--brokers',
            help='Topic to publish to, env variable KAFKA_BROKERS',
            default='kafka:9092')
    parser.add_argument(
            '--topic',
            help='Topic to publish to, env variable KAFKA_TOPIC',
            default='cord-19-nlp')
    cmdline_args = parse_args(parser)
    main(cmdline_args)
    logging.info('exiting')
