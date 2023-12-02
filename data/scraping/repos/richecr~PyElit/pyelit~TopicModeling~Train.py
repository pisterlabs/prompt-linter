import os
import nltk
import random
import spacy
import gensim
import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim.test.utils import datapath
from sklearn.model_selection import train_test_split


def main_cross_val():
    # Useing cross-validation
    # Configuring libraries for better results.
    np.random.seed(2018)
    nltk.download("wordnet")
    nlp = spacy.load("pt_core_news_sm")

    # PREPARING FILES.
    data = pd.read_csv("../../dados/textos_limpos.csv")
    data.drop_duplicates(["texto"], inplace=True)
    texts = data["texto"]
    texts = [str(texto) for text in texts]
    cross = kfoldcv(texts)
    print("Total: ", len(texts))

    for train_tests in cross:
        train = train_tests[0]
        test = train_tests[1]

        processed_docs = [t.split() for t in train]
        processed_test = [t.split() for t in test]

        print("Train: ", len(processed_docs))
        print("Tests: ", len(processed_test))

        # Creating dictionary of words.
        dictionary = gensim.corpora.Dictionary(processed_docs)

        # Gensim Filter Extremes
        # Filter tokens that appear in less than 15 documents.
        # or in more than 0.5 documents (fraction of the total corpus size).
        # After these two steps, keep only 100000.
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        # Bag of Words.
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Using TF-IDF.
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]

        # Creating and training the model.
        lda_model_tfidf = gensim.models.LdaMulticore(
            corpus_tfidf,
            num_topics=4,
            id2word=dictionary,
            passes=10,
            workers=4,
            alpha=0.01,
            eta=0.9,
        )

        coherence_model(lda_model_tfidf, processed_test, corpus_tfidf, dictionary)


def coherence_model(lda_model_, tests, corpus, dictionary):
    coherence_model_lda = CoherenceModel(
        model=lda_model_,
        texts=tests,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print("\nCoherence Score LDAModelTfIdf: ", coherence_lda)


def kfoldcv(dados, k=6, seed=42):
    size = len(dados)
    subset_size = round(size / k)
    random.Random(seed).shuffle(dados)
    subsets = [dados[x : x + subset_size] for x in range(0, len(dados), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                for ss in subset:
                    train.append(ss)
        kfolds.append((train, test))

    return kfolds


def main():
    # Configuring libraries for better results.
    np.random.seed(2018)
    nltk.download("wordnet")
    nlp = spacy.load("pt_core_news_sm")

    # PREPARING FILES.
    data = pd.read_csv("../../dados/textos_limpos.csv")
    data.drop_duplicates(["texto"], inplace=True)

    processed_docs = data["texto"].map(lambda text: text.split())
    print(processed_docs[:10])

    # Creating dictionary of words.
    dictionary = gensim.corpora.Dictionary(processed_docs)

    # Gensim Filter Extremes
    # Filter tokens that appear in less than 15 documents
    # or in more than 0.5 documents (fraction of the total corpus size).
    # After these two steps, keep only 100000.
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # Bag of Words.
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Using TF-IDF.
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # Creating and training the model.
    # lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4,
    #                                        id2word=dictionary, passes=10,
    #                                        workers=4, alpha=0.01, eta=0.9)
    # slda_model.save("./modelo/meu_lda_model")

    ROOT = os.path.abspath(os.path.dirname(__file__))
    fname = datapath(ROOT + "/modelo/meu_lda_model")
    model = gensim.models.LdaMulticore.load(fname=fname)

    def coherence_model(lda_model_, processed_docs, corpus_tfidf, dictionary):
        coherence_model_lda = CoherenceModel(
            model=lda_model_,
            texts=processed_docs,
            corpus=corpus_tfidf,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_lda = coherence_model_lda.get_coherence()
        print("\nCoherence Score LDAModelTfIdf load: ", coherence_lda)

    coherence_model(model, processed_docs, corpus_tfidf, dictionary)

    def compute_num_steps(dct, corpus_tfidf, texts, limit, start, step):
        coherence_values = []
        model_list = []
        for passes in range(start, limit, 1):
            model = gensim.models.LdaMulticore(
                corpus_tfidf,
                num_topics=4,
                id2word=dictionary,
                passes=passes,
                workers=4,
                alpha=0.01,
                eta=0.9,
            )
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model,
                texts=texts,
                corpus=corpus_tfidf,
                dictionary=dct,
                coherence="c_v",
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    model_list, coherence_values = compute_num_steps(
        dictionary, corpus_tfidf, processed_docs, 13, 9, 1
    )

    limit = 13
    start = 9
    step = 1
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Steps")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc="best")
    plt.show()

    for m, cv in zip(x, coherence_values):
        print("Step =", m, " has Coherence Value of", round(cv, 4))


main()
# main_cross_val()
