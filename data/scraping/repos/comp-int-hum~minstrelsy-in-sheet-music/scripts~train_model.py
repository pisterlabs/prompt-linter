import csv
import re
import logging
import random
import pickle
import json
import glob
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.parsing.preprocessing as gpp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", dest='inputs', help="input file" )
parser.add_argument("--topic_num", dest= "topic_numbers", help = "number of topics")
parser.add_argument("--output_file", dest="output_file", help="output file")
parser.add_argument("--chunk_size", dest = "chunk_size", default = 500, help = "size of topic chunks")

args = parser.parse_args()
data = args.inputs
no_topics = args.topic_numbers
chunk_size = args.chunk_size
output_file = args.output_file

logging.basicConfig(level=logging.INFO)

minimum_characters = 3
groupwise_topic_counts = {}
tokenizer = RegexpTokenizer(r'\w+')

holder_list = []
output_list = []
levy_data = []

with open(data, 'rt') as json_obj:
    for x in json_obj:
        line = json.loads(x)
        levy_data.append(line)
    for row in levy_data:
        holder_list.append(row["full_text"])
        
for x in holder_list:
    tokenized = gpp.split_on_space(
            gpp.strip_short(
                gpp.remove_stopwords(
                    gpp.strip_non_alphanum(
                        x
                    ),
                ),
            minsize=minimum_characters
            )
        )
    ls = []
    for y in tokenized:
       y = y.lower()
       ls.append(y)
    output_list.append(ls)
   
    
dct = Dictionary(documents = output_list)
dct.filter_extremes(no_below=5, no_above=0.7)
temp = dct[0]
#dct.save("work/dictionary_{}_topics.gensim".format(no_topics))
corpus = [dct.doc2bow(doc) for doc in output_list]
model = LdaModel(
corpus, 
    num_topics = no_topics, 
    id2word=dct, 
    alpha="auto", 
    eta="auto", 
    iterations=50, 
    passes=25, 
    eval_every=None, 
    chunksize= chunk_size
)

with open(output_file, "wb") as ofd:
    ofd.write(pickle.dumps(model))





    
