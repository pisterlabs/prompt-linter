import os
from dataclasses import dataclass

from gensim.models import CoherenceModel
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable

import itertools as it
from pprint import pprint

import pandas as pd
import gensim.corpora as corpora
import gensim

import numpy as np
from sklearn.neighbors import KDTree


# Configuration parsing
@dataclass
class Config(Serializable):
    input_dataset: str
    output_model: str
    n_topics: int = 100
    n_iters: int = 6000
    n_passes: int = 100
    decay: float = 0.0005
    seed: int = 100
    explore: bool = True
    explore_query_id: int = 0


parser = ArgumentParser()
parser.add_arguments(Config, dest="config")
args = parser.parse_args()
config: Config = args.config

print('Loading dataset')
df = pd.read_parquet(config.input_dataset)
df = df[df['terms'].notnull() & df['terms'].str.len() > 0]
df.reset_index(inplace=True)

print('Extracting words and converting dataset to bow')
tags = set(it.chain.from_iterable(df['terms']))
id2word = corpora.Dictionary([tags])
corpus = [id2word.doc2bow(tags) for i, tags in df['terms'].iteritems()]


def train_model(n_topics):
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_topics,
        iterations=config.n_iters,
        eval_every=50,
        passes=config.n_passes,
        decay=config.decay,
        random_state=config.seed,
    )

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    preplexity = lda_model.log_perplexity(corpus)
    # Coherence score: Higher is better
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=df['terms'].tolist(), dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return lda_model, preplexity, coherence_lda


def explore_query(lda_model):
    print(f'Querying {config.explore_query_id}')
    topics = {k: v for (k, v) in lda_model.print_topics(num_topics=1000)}
    doc_lda = lda_model.get_document_topics(corpus, 0.)
    doc_data = np.array(doc_lda)[:, :, 1]
    topic_data = np.nan_to_num(doc_data / doc_data.sum(axis=1).reshape(-1, 1))
    kd = KDTree(topic_data, metric='euclidean')

    query_index = df.index[df.id == config.explore_query_id].to_list()

    # Print query topics
    print('Query Topics')
    for i, v in sorted([(i, v) for i, v in enumerate(topic_data[query_index[0], :]) if v > 0.01], key=lambda t: t[1]):
        print(f'Value {v}, Topic: {topics[i]}')

    # Print query results
    print('Matches')
    if len(query_index) > 0:
        dd, ii = kd.query([topic_data[query_index[0], :]], 20)
        result = df.iloc[ii[0]].copy()
        result['dist'] = dd[0].tolist()
        print(result[['id', 'title', 'dist']])


os.makedirs(os.path.join('models', config.output_model))
print('Trying different topic values')
results = []
if config.explore:
    for n_topics in [5, 8, 20, 40, 80, 100, 120]:
        print(f'- Explore: n_topics={n_topics}')
        model, perp, coh = train_model(n_topics)
        print(f'Result: perp={perp}, coh={coh}')
        results.append((n_topics, model, perp, coh))
        explore_query(model)
        model.save(os.path.join('models', config.output_model, f'lda_ntopics_{n_topics}'))
