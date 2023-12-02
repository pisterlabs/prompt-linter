import nltk
import pandas as pd
import numpy as np
import re
import codecs
import json

import re
import numpy as np
import pandas as pd
from pprint import pprint
import _pickle as pickle
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
from gensim.models.ldamulticore import LdaMulticore
# spacy for lemmatization
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords, names

nlp = spacy.load('en')
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def get_text(rev):
    return rev[0] if str(rev)!='nan' else ''

def clean_text(text):
    import re, string
    regex = re.compile('[%s]' % re.escape(string.punctuation.replace('.','')))
    text = regex.sub('', text)
    text = re.sub('/', ' ', text)
#     text = re.sub('\s+', ' ', text)
#     text = re.sub("\'", "", text)
    return text.lower()

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def spell_correction(text):
    from textblob import TextBlob
    return TextBlob(text).correct()

def prepare_text_for_lda(text):
    tokens = tokenize(text)
#     tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



# %%time
# alldata = pd.DataFrame(pd.concat([nydata['FullReview'],calidata['FullReview']],ignore_index=True))
# alldata.to_json('all_reviews.json', orient='columns')
alldata = pd.read_json('all_reviews.json',orient='columns', encoding='utf-8')
# alldata.sample()

# helper functions for text preprocessing & LDA modeling:

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space or token.like_num or token.is_digit

def line_review(filename):
    """
    generator function to read in reviews from Pandas Series
    and un-escape the original line breaks in the text
    """

    #with codecs.open(filename, encoding='utf_8') as f:
    for review in filename:
        yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """

    for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=10):

        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    # load finished dictionary from disk
    trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')

    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

def reviewPreProcess(text):
    """
    Takes Pandas series as input,
    consisting of one review as
    text string per row
    """
    from tqdm import tqdm
    # lemmatized_sentence_corpus generator loops over original review text, segments the reviews into individual sentences
    # and normalizes text. Writes data to new file with one normalized sentence per line:


    # with codecs.open('./models2/unigram_sentences_p5.txt', 'a', encoding='utf_8') as f:
    #     for sentence in tqdm(lemmatized_sentence_corpus(text)):
    #         # print(sentence)
    #         f.write(sentence + '\n')
    # f.close()
    # Create object to stream unigram sentences from disk, rather than hold in memory:
    # unigram_sentences = LineSentence('./models2/unigram_sentences.txt')
    #
    # # Train phrase model to link individual words into two-word phrases:
    # bigram_model = Phrases(unigram_sentences)
    # bigram_model.save('./models2/bigram_model.txt')
    #
    # # Apply trained bigram phrase model to the review sentences data:
    # with codecs.open('./models2/bigram_sentences.txt', 'w', encoding='utf_8') as f:
    #     for unigram_sentence in tqdm(unigram_sentences):
    #         bigram_sentence = u' '.join(bigram_model[unigram_sentence])
    #         f.write(bigram_sentence + '\n')
    #
    # # Create object to stream bigram sentences from disk, rather than hold in memory:
    # bigram_sentences = LineSentence('./models2/bigram_sentences.txt')
    #
    # # Train second-order phrase model to to generate trigrams:
    # trigram_model = Phrases(bigram_sentences)
    # trigram_model.save('./models2/trigram_model.txt')
    #
    # # Apply trained second-order phrase model to our first-order transformed sentences and write to a new file:
    # with codecs.open('./models2/trigram_sentences.txt', 'w', encoding='utf_8') as f:
    #     for bigram_sentence in tqdm(bigram_sentences):
    #         trigram_sentence = u' '.join(trigram_model[bigram_sentence])
    #         f.write(trigram_sentence + '\n')

    # Run complete text of the reviews through a pipeline that applies text normalization and phrase models.
    # Also remove stopwords and write transformed text to a new file, one review per line:
#     bigram_model = Phrases.load('./models2/bigram_model.txt')
#     trigram_model = Phrases.load('./models2/trigram_model.txt')
#     with codecs.open('./models2/tri/trigram_transformed_reviews_p6.txt', 'a', encoding='utf_8') as f:
#         for parsed_review in tqdm(nlp.pipe(line_review(text),
#                                         batch_size=10000, n_threads=10)):
#
#             # lemmatize the text, removing punctuation and whitespace
#             unigram_review = [token.lemma_ for token in parsed_review
#                                 if not punct_space(token)]
#
#             # apply the first-order and second-order phrase models
#             bigram_review = bigram_model[unigram_review]
#             trigram_review = trigram_model[bigram_review]
#
# #             common_terms = ['order', 'come', 'bad', 'good', \
# #                             'place', 'time', '\'s'] #'service',
#
#             # remove any remaining stopwords
#             trigram_review = [term for term in trigram_review
#                                 if term not in spacy.lang.en.stop_words.STOP_WORDS]
# #             trigram_review = [term for term in trigram_review
# #                                 if term not in common_terms]
#
#             # write the transformed review as a line in the new file
#             trigram_review = u' '.join(trigram_review)
#             f.write(trigram_review + '\n')


    # Learn full vocabulary of corpus to be modeled, using gensim's Dictionary class.  Stream
    # reviews off of disk using LineSentence:
    trigram_reviews = LineSentence('./models2/trigram_transformed_reviews.txt')

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_reviews)

    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=3, no_above=0.4)
    trigram_dictionary.compactify()

    trigram_dictionary.save('./models2/trigram_dict_all.dict')

    return 1#bigram_model, trigram_model, trigram_dictionary


# def LDA_Model(topics, cores=11):
#     """
#     Topics represents desired LDA topics,
#     cores should be physical cores minus one.
#     Both should be integers.
#     """
#
#     # load finished dictionary from disk
#     trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')
#
#     # generate bag-of-words representations for
#     # all reviews and save them as a matrix
#     MmCorpus.serialize('./models2/trigram_bow_corpus.nm',
#                         trigram_bow_generator('./models2/trigram_transformed_reviews.txt'))
#
#     # load finished bag-of-words corpus from disk
#     trigram_bow_corpus = MmCorpus('./models2/trigram_bow_corpus.nm')
#
#
#     # Pass the bag-of-words matrix and Dictionary from previous steps to LdaMulticore as inputs,
#     # along with the number of topics the model should learn
#
#     # workers => sets the parallelism, and should be
#     # set to your number of physical cores minus one
#     lda = LdaMulticore(trigram_bow_corpus,
#                        num_topics=topics,
#                        id2word=trigram_dictionary,
#                        workers=cores)
#
#     lda.save('./models2/lda_model')
#
#     # load the finished LDA model from disk
#     #lda = LdaMulticore.load('./models/lda_model_neg')
#
#     return trigram_bow_corpus, lda

def guidedLDA_Model(topics, cores=11):
    """
    Topics represents desired LDA topics,
    cores should be physical cores minus one.
    Both should be integers.
    """

    # load finished dictionary from disk
    trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')

    # generate bag-of-words representations for
    # all reviews and save them as a matrix
    MmCorpus.serialize('./models2/trigram_bow_corpus.nm',
                        trigram_bow_generator('./models2/trigram_transformed_reviews.txt'))

    # load finished bag-of-words corpus from disk
    trigram_bow_corpus = MmCorpus('./models2/trigram_bow_corpus.nm')


    # Pass the bag-of-words matrix and Dictionary from previous steps to LdaMulticore as inputs,
    # along with the number of topics the model should learn

    # workers => sets the parallelism, and should be
    # set to your number of physical cores minus one
    lda = LdaMulticore(trigram_bow_corpus,
                       num_topics=topics,
                       id2word=trigram_dictionary,
                       workers=cores)

    lda.save('./models2/lda_model')

    # load the finished LDA model from disk
    #lda = LdaMulticore.load('./models/lda_model_neg')

    return trigram_bow_corpus, lda
# %%time
# N = len(alldata)
# ii=800000
# ff=ii+20000
# while ff<N:
#     aa = reviewPreProcess(alldata['FullReview'][ii:ff])
#     ii=ff
#     ff=ii+20000
#     print(ff)
# else:
#     aa = reviewPreProcess(alldata['FullReview'][ii:N])
d = reviewPreProcess(alldata['FullReview'])
# bigram_model, trigram_model, trigram_dictionary = reviewPreProcess(alldata['FullReview'])
trigram_bow_corpus, lda = LDA_Model(15)
import pickle
trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')
trigram_bow_corpus = MmCorpus('./models2/trigram_bow_corpus.nm')
lda = LdaMulticore.load('./models2/lda_model')

LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dictionary)

# Save pre-prepared pyLDAvis data to disk:
with open('./models2/ldavis_prepared', 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk:
with open('./models2/ldavis_prepared', 'rb') as f:
    LDAvis_prepared = pickle.load(f)

# pyLDAvis.display(LDAvis_prepared)
pyLDAvis.save_html(LDAvis_prepared,'./models2/lda.html')
