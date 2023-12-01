import json
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

from elasticsearch import Elasticsearch
from elasticsearch import helpers


# Configuration parsing
@dataclass
class Config(Serializable):
    input_dataset: str
    input_model: str
    index_name: str
    index_mapping: str
    n_topics: int = 100


parser = ArgumentParser()
parser.add_arguments(Config, dest="config")
args = parser.parse_args()
config: Config = args.config

print('Loading dataset')
df = pd.read_parquet(config.input_dataset)
df = df[df['terms'].notnull() & df['terms'].str.len() > 0]
df.reset_index(inplace=True)



print('Loading existing model')
lda_model = gensim.models.LdaMulticore.load(
    os.path.join('models', config.input_model, f'lda_ntopics_{config.n_topics}')
)

print('Extracting words and converting dataset to bow')
id2word = lda_model.id2word
corpus = [id2word.doc2bow(tags) for i, tags in df['terms'].iteritems()]

print('Transforming documents')
doc_lda = lda_model.get_document_topics(corpus, 0.)
topic_data = np.array(doc_lda)[:, :, 1]

print('Transforming pandas dataset')
documents = df.copy()
documents['topics'] = topic_data.tolist()

example = documents.iloc[0]
for col in documents.columns:
    if isinstance(example[col], np.ndarray):
        print(f'Converting column {col} to list')
        documents[col] = documents[col].map(lambda x: x.tolist())

print('Building elasticsearch index')
es = Elasticsearch(hosts=['nas:9200'], http_compress=True, timeout=30)
mapping = json.loads(open(config.index_mapping, 'r').read())
mapping['settings'] = {"index.knn": True}
mapping['mappings']['properties']['terms'] = {"type": "keyword"}
mapping['mappings']['properties']['topics'] = {"type": "knn_vector", "dimension": config.n_topics}
es.indices.create(index=config.index_name, body=mapping)


def doc_generator(df):
    for document in df.to_dict(orient='records'):
        yield {
            "_index": config.index_name,
            "_type": "_doc",
            "_id": f"{document['id']}",
            "_source": document,
        }


print('Pushing elasticsearch data')
if es.indices.exists(index=config.index_name):
    es.delete_by_query(index=[config.index_name], body={"query": {"match_all": {}}})
helpers.bulk(es, doc_generator(documents))
