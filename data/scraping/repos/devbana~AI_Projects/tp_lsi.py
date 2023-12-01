
import os
import gensim
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from gensim.utils import lemmatize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.parsing.preprocessing import strip_numeric, strip_short, strip_multiple_whitespaces,strip_non_alphanum,strip_punctuation,strip_tags,preprocess_string
import pandas as pd
from gensim import similarities
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from pprint import pprint



#read the data
corpus_dir = 'https://raw.githubusercontent.com/Ramaseshanr/anlp/master/corpus/bbc-text.csv'
df_corpus = pd.read_csv(corpus_dir, names=['category', 'text'])
corpus = df_corpus['text'].values.tolist()
corpus = corpus[1:]
my_filter = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short, stem_text
]


def preprocessing(corpus):
    for document in corpus:
        doc = strip_numeric(document)
        doc = remove_stopwords(doc)
        doc = strip_short(doc, 3)
        #doc = stem_text(doc)
        doc = strip_punctuation(doc)
        strip_tags(doc)
        yield gensim.utils.tokenize(doc, lower=True)


texts = preprocessing(corpus)
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, keep_n=25000)

doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in preprocessing(corpus)]
tfidf = models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]

print('LSI Model')
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)  # initialize an LSI transformation
topics = lsi.print_topics(num_topics=5, num_words=25)
for i in topics:
    print(i[0])
    print(i[1])


