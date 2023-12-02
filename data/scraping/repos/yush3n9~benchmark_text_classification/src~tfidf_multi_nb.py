#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import defaultdict

import gensim.matutils as matutils
import numpy as np
import spacy
from gensim import corpora, models
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

nlp = spacy.load('en')
from src.utils.newsgroup_dataprovider import TwentyNewsgroup
import src.utils.text_preprocessing as tp
import pickle
from gensim.models import CoherenceModel

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://arxiv.org/pdf/1607.01759.pdf
# https://statsbot.co/blog/text-classifier-algorithms-in-machine-learning/


random_state = 42
n_features = 500


def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]


# ----------------------
# Fetch train dataset:
# ----------------------
dp = TwentyNewsgroup(categories=[
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'])

# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english', max_features=n_features)  # accuracy 83.75 83,83
# vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))  # 83.72 83.83


# ---------------------
# Reduce size of train manuelly
######################
X_train_small, y_train_small = list(), list()
# size_category = 50
# cat_counter = dict()
# for _x, _y in zip(dp.fetch_dataset_train().data, dp.fetch_dataset_train().target):
#     if _y in cat_counter:
#         if cat_counter[_y] < size_category:
#             X_train_small.append(_x)
#             y_train_small.append(_y)
#             cat_counter[_y] += 1
#         else:
#             # print('Category is full ...')
#             continue
#     else:
#         X_train_small.append(_x)
#         y_train_small.append(_y)
#         cat_counter[_y] = 1

# X_train = vectorizer.fit_transform(dp.fetch_dataset_train().data)

n_samples = 1000
X_train_small = dp.fetch_dataset_train().data[:n_samples]
y_train_small = dp.fetch_dataset_train().target[:n_samples]

X_train = vectorizer.fit_transform(X_train_small)
X_test = vectorizer.transform(dp.fetch_dataset_test().data)

# y_train = dp.fetch_dataset_train().target
y_train = y_train_small
# y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
# y_test = to_categorical(np.asarray(y_test))

clf = MultinomialNB(alpha=0.05)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
print('MNB Accuracy: %f' % (100 * accuracy_score(predicted, y_test)))

# -------------------------
# LDA as features
# -------------------------
import os

num_topics = 40
dictionary_path = 'dict'
corpus_path = 'corpus'
lda_path = 'lda'
token_path = 'tokens'


def text_preprocessing(documents):
    # https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
    # stop words removal
    # prunduction
    # word frequency filter
    # POS filter
    cleaned_documents = [tp.clean_single_doc(doc) for doc in tqdm(documents, 'Text cleaning')]
    # nlp_docs = [nlp(clean_text) for clean_text in tqdm(cleaned_documents, 'Spacy nlu processing')]
    # tokenized_docs = [tp.filter_single_nlp_doc(doc) for doc in tqdm(nlp_docs, 'nlp POS filtering')]

    import nltk
    tokenized_docs = [nltk.word_tokenize(doc) for doc in tqdm(cleaned_documents, 'nltk tokenizing')]

    with open('{}/{}'.format(token_path, len(documents)), 'wb') as f:
        pickle.dump(tokenized_docs, f)
    return tokenized_docs


def load_tokens(documents):
    if os.path.isfile('{}/{}'.format(token_path, len(documents))):
        with open('{}/{}'.format(token_path, len(documents)), 'rb') as f:
            return pickle.load(f)
    else:
        return text_preprocessing(documents)


def tokenize(documents):
    tokens_all = [word_tokenize(doc) for doc in tqdm(documents, 'tokenize 1')]
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in tokens_all:
        for token in text:
            frequency[token] += 1

    tokens_all = [[token for token in text if frequency[token] > 1]
                  for text in tokens_all]
    return tokens_all


def build_dictionary(tokens):
    dictionary = corpora.Dictionary(tokens)
    dictionary.save('{}/{}.dict'.format(dictionary_path, len(tokens)))
    return dictionary


def load_dictionary(tokens):
    if os.path.isfile('{}/{}.dict'.format(dictionary_path, len(tokens))):
        return corpora.Dictionary.load('{}/{}.dict'.format(dictionary_path, len(tokens)))
    else:
        return build_dictionary(tokens)


def build_corpus(dictionary, tokens):
    corpus = [dictionary.doc2bow(text) for text in tokens]
    corpora.MmCorpus.serialize('{}/corpus{}.mm'.format(corpus_path, len(tokens)), corpus)
    return corpus


def load_corpus(dictionary, tokens):
    if os.path.isfile('{}/corpus{}.mm'.format(corpus_path, len(tokens))):
        return corpora.MmCorpus('{}/corpus{}.mm'.format(corpus_path, len(tokens)))
    else:
        return build_corpus(dictionary, tokens)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def load_ldamodel(tokens, dictionary, corpus):
    if os.path.isfile('{}/lda{}.model'.format(lda_path, num_topics)):
        return models.LdaModel.load('{}/lda{}.model'.format(lda_path, num_topics), mmap='r')
    else:
        gensim_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
        gensim_lda.save('{}/lda{}.model'.format(lda_path, num_topics))
        return gensim_lda


def use_gensim_lda():
    # tokens = tokenize(dp.fetch_dataset_train().data)
    # tokens = tokenize(X_train_small)
    # print(tokens[0])

    # tokens2 = text_preprocessing(X_train_small)
    # tokens = load_tokens(X_train_small)
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    tokens = [[t for t in simple_preprocess(text) if t not in STOPWORDS] for text in X_train_small]

    dictionary = load_dictionary(tokens)
    corpus = load_corpus(dictionary, tokens)
    gensim_lda = load_ldamodel(tokens, dictionary, corpus)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=gensim_lda, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    for t in gensim_lda.print_topics(num_topics, 10):
        print(t)

    # pprint(gensim_lda.print_topics())

    # print(tokens_all)

    # lda = LDA(n_components=num_topics)
    # lda.fit(X_train)

    X_train_bow = [dictionary.doc2bow(word_tokenize(train_doc)) for train_doc in
                   tqdm(X_train_small, 'Build corpus on trainset')]

    X_test_bow = [dictionary.doc2bow(word_tokenize(test_doc)) for test_doc in
                  tqdm(dp.fetch_dataset_test().data, 'Build corpus on testset')]

    training_features = gensim_lda[tqdm(X_train_bow, 'topic modelling on trainset')]
    training_features = matutils.corpus2dense(training_features, num_terms=num_topics)
    training_features = training_features.T

    print('{}+{}'.format(training_features.shape, X_train.shape))
    combo_train_feature = np.append(training_features, X_train.todense(), 1)
    print('{}+{}={}'.format(training_features.shape, X_train.shape, combo_train_feature.shape))

    testing_features = gensim_lda[tqdm(X_test_bow, 'topic modelling on testset')]
    testing_features = matutils.corpus2dense(testing_features, num_terms=num_topics)
    testing_features = testing_features.T
    combo_test_feature = np.append(testing_features, X_test.todense(), 1)

    # print(len(training_features))

    # lsvc = LinearSVC(C=0.05, penalty="l2", dual=False).fit(X_train, y_train)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_train = model.transform(X_train)

    # model = SelectKBest(chi2, k=5).fit(training_features, y_train)
    # training_features = model.transform(training_features)
    # testing_features = model.transform(testing_features)
    # print(len(training_features))

    # MNB works not good on lda
    # Try SVM
    # clf = LinearSVC(C=1, penalty="l1", dual=False, tol=1e-4)

    clf.fit(training_features, y_train_small)
    predicted = clf.predict(testing_features)
    print('gensim lda NB Accuracy: %f' % (100 * accuracy_score(predicted, y_test)))




def use_pypi_lda():
    import lda
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=n_features, stop_words='english')

    bow_train = vectorizer.fit_transform(
        tqdm([tp.clean_single_doc(doc) for doc in tqdm(X_train_small, 'Text cleaning')]))
    bow_test = vectorizer.transform(
        tqdm([tp.clean_single_doc(doc) for doc in tqdm(dp.fetch_dataset_test().data, 'Text cleaning')]))

    model = lda.LDA(n_topics=num_topics, random_state=random_state).fit(bow_train)

    topic_summaries = []
    topic_word = model.topic_word_  # all topic words
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(10 + 1):-1]  # get!
        topic_summaries.append(' '.join(topic_words))  # append!
    for t in topic_summaries:
        print(t)

    combo_train = np.append(model.transform(bow_train), X_train.todense(), 1)
    combo_test = np.append(model.transform(bow_test), X_test.todense(), 1)

    clf.fit(combo_train, y_train_small)

    predicted = clf.predict(combo_test)
    print('lda NB Accuracy: %f' % (100 * accuracy_score(predicted, y_test)))


    clf.fit(bow_train, y_train_small)
    predicted = clf.predict(bow_test)
    print('lda NB pure Accuracy: %f' % (100 * accuracy_score(predicted, y_test)))

if __name__ == '__main__':
    use_gensim_lda()
    use_pypi_lda()
