import pandas as pd
from gensim.test.utils import datapath

import re
import sys
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LsiModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import ast
import warnings
warnings.filterwarnings("ignore")

def main(file_name):
    data = pd.read_csv(file_name, error_bad_lines=False);
    doc_clean = data['tokens_final']
    doc_clean = map(str, doc_clean) #For mongo
    doc_clean1 = [ast.literal_eval(doc) for doc in doc_clean]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(doc_clean1)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean1]
    return doc_term_matrix

def LDA(doc_term_matrix):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.LdaMulticore
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=25, id2word = dictionary, passes=50, workers=4)
    ldamodel.save("ldamodel_sample")
    # Load a potentially pretrained model from disk.
    ldamodel = gensim.models.LdaMulticore.load("ldamodel_sample")
    pprint(ldamodel.print_topics(num_topics=15, num_words=5))
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis,fileobj='visuals.html')
    
def LSI(doc_term_matrix):
    # Running and Trainign LDA model on the document term matrix.
    lsimodel = LsiModel(doc_term_matrix, num_topics=25, id2word = dictionary, decay=0.5)
    lsimodel.save("lsimodel")
    pprint(lsimodel.print_topics(-1))
    
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def joiner(X):
    return ' '.join(X)

def generate_keywords():
    doc_clean_joiner = list(map(joiner,doc_clean1))    
    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix, texts=doc_clean_joiner)
    df_topic_sents_keywords.to_csv('keywords.csv')

if __name__ == '__main__':
    token_file = sys.argv[1]
    dtm = main(token_file)
    LDA(dtm)
    LSI(dtm)
    generate_keywords()
    
