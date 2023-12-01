import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import pickle

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class gensim_topic_models():


    def __init__(self, data=None, bigram_min_count=5, bigram_threshold=100):
        if data is not None:
            # data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

            # Remove new line characters
            data = [re.sub('\s+', ' ', sent) for sent in data]

            # Remove distracting single quotes
            data = [re.sub("\'", "", sent) for sent in data]

            self.data_words = list(self.sent_to_words(data))

            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(self.data_words, min_count=bigram_min_count, threshold=bigram_threshold)  # higher threshold fewer phrases.
            # trigram = gensim.models.Phrases(bigram[self.data_words], threshold=bigram_threshold)

            # Faster way to get a sentence clubbed as a trigram/bigram
            self.bigram_mod = gensim.models.phrases.Phraser(bigram)
            # self.trigram_mod = gensim.models.phrases.Phraser(trigram)



    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    # def make_trigrams(self, texts):
    #     return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    def preprocess(self):
        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(self.data_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        return data_lemmatized

    def create_dictionary_and_term_document_freq(self, data_lemmatized):
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        with open("Echo networks/Topic model/gensim_tm/corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        with open("Echo networks/Topic model/gensim_tm/id2word.pkl", "wb") as f:
            pickle.dump(id2word, f)
        with open("Echo networks/Topic model/gensim_tm/texts.pkl", "wb") as f:
            pickle.dump(texts, f)
        return corpus, id2word, texts


    def create_gensim_lda_model(self, K, corpus, id2word, texts):
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=K,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)

        # save model
        lda_model.save("Echo networks/Topic model/gensim_tm/gensim_lda_model_{}_topics.model".format(K))
        with open("gensim_lda_{}_topics.txt".format(K), "w") as f:

            # Print the Keyword in the 10 topics
            #pprint(lda_model.print_topics())
            # doc_lda = lda_model[corpus]
            for topic in lda_model.print_topics():
                f.write(str(topic))

            # Compute Perplexity
            print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
            f.write('\nPerplexity: ' + str(lda_model.log_perplexity(corpus)))
            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: {}'.format(coherence_lda))
            f.write('\nCoherence Score: {}'.format(coherence_lda))

















# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# pyLDAvis.display(vis)




# # later on, load trained model from file
# model =  gensim.models.LdaModel.load('lda.model')
#
# # print all topics
# model.show_topics(topics=200, topn=20)
#
# # print topic 28
# model.print_topic(109, topn=20)
#
# # another way
# for i in range(0, model.num_topics-1):
#     print model.print_topic(i)
#
# # and another way, only prints top words
# for t in range(0, model.num_topics-1):
#     print 'topic {}: '.format(t) + ', '.join([v[1] for v in model.show_topic(t, 20)])




######################################################
# import spacy
# spacy.load('en')
# from spacy.lang.en import English
# parser = English()
# def tokenize(text):
#     lda_tokens = []
#     tokens = parser(text)
#     for token in tokens:
#         if token.orth_.isspace():
#             continue
#         elif token.like_url:
#             lda_tokens.append('URL')
#         elif token.orth_.startswith('@'):
#             lda_tokens.append('SCREEN_NAME')
#         else:
#             lda_tokens.append(token.lower_)
#     return lda_tokens
#
# import nltk
#
# nltk.download('wordnet')
# from nltk.corpus import wordnet as wn
#
#
# def get_lemma(word):
#   lemma = wn.morphy(word)
#   if lemma is None:
#     return word
#   else:
#     return lemma
#
#
# from nltk.stem.wordnet import WordNetLemmatizer
#
#
# def get_lemma2(word):
#   return WordNetLemmatizer().lemmatize(word)
# # nltk.download('stopwords')
# en_stop = set(nltk.corpus.stopwords.words('english'))
# def prepare_text_for_lda(text):
#   tokens = tokenize(text)
#   tokens = [token for token in tokens if len(token) > 4]
#   tokens = [token for token in tokens if token not in en_stop]
#   tokens = [get_lemma(token) for token in tokens]
#   return tokens
# import random
# text_data = []
# with open('dataset.csv') as f:
#     for line in f:
#         tokens = prepare_text_for_lda(line)
#         if random.random() > .99:
#             print(tokens)
#             text_data.append(tokens)
# from gensim import corpora
# dictionary = corpora.Dictionary(text_data)
# corpus = [dictionary.doc2bow(text) for text in text_data]
# import pickle
# pickle.dump(corpus, open('corpus.pkl', 'wb'))
# dictionary.save('dictionary.gensim')
# import gensim
# NUM_TOPICS = 5
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
# ldamodel.save('model5.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
# new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
# new_doc = prepare_text_for_lda(new_doc)
# new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# print(ldamodel.get_document_topics(new_doc_bow))
#
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
# ldamodel.save('model3.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
#
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
# ldamodel.save('model10.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
# corpus = pickle.load(open('corpus.pkl', 'rb'))
# lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
# # import pyLDAvis.gensim
# # lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# # pyLDAvis.display(lda_display)
#
#
# # lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')
# # lda_display3 = pyLDAvis.gensim.prepare(lda3, corpus, dictionary, sort_topics=False)
# # pyLDAvis.display(lda_display3)
# print("1")