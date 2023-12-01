import pickle
import random
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
import gensim.models as models
from gensim.models import CoherenceModel
from data_preparation import remove_low_high_frequent_words, get_tfidf, extract_important_words_tfidf


def save_train_and_test(data):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    dictionary = get_tfidf(documents)["index2word"]

    corpus = list(corpus)

    random.shuffle(corpus)
    train = corpus[:18000]
    test = corpus[18000:]

    with open('data/train_corpus.data', 'wb') as file:
        print("...Saving training corpus into local binary file...")
        pickle.dump(train, file)

    with open('data/test_corpus.data', 'wb') as file:
        print("...Saving test corpus into local binary file...")
        pickle.dump(test, file)

    with open('data/common_dictionary.data', 'wb') as file:
        print("...Saving common dictionary for train and test corpus into binary file...")
        pickle.dump(dictionary, file)


def perplexity(train_corpus, test_corpus, dictionary, limit, start=2, step=1):
    perplexity_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=num_topics, eta=0.3)
        model_list.append(model)

        perplexity_values.append(model.bound(test_corpus))

    return model_list, perplexity_values


def coherence(train_corpus, test_corpus, dictionary, limit, start=2, step=1):
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=num_topics, eta=0.3)
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, dictionary=dictionary, corpus=test_corpus, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def main():
    # # ++++ Loading train corpus and training LDA model with it ++++ #
    # with open('data/common_dictionary.data', 'rb') as file:
    #     dictionary = pickle.load(file)
    #
    # with open('data/train_corpus.data', 'rb') as file:
    #     train_corpus = pickle.load(file)
    #
    # model = models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=5, eta=0.3)
    # print('{} {}'.format('length of the dictionary is: ', len(dictionary)))
    #
    # visual = pyLDAvis.gensim.prepare(model, train_corpus, dictionary)
    # pyLDAvis.save_html(visual, 'visual/lda_visual.html')
    # # ------------------------------------------------------------- #

    # ++++ LDA coherence score on test data ++++ #
    with open('data/common_dictionary.data', 'rb') as file:
        dictionary = pickle.load(file)

    with open('data/train_corpus.data', 'rb') as file:
        train_corpus = pickle.load(file)

    with open('data/test_corpus.data', 'rb') as file:
        test_corpus = pickle.load(file)

    limit = 15
    start = 2
    step = 1
    model_list, coherence_values = coherence(train_corpus, test_corpus, dictionary, limit=limit, start=start, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    # plt.legend("coherence_score", loc='best')
    plt.show()
    # ------------------------------------------------------------- #


if __name__ == '__main__':
    main()
