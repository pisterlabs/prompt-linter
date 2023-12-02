import gensim.models as models
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from data_preparation import remove_low_high_frequent_words, get_tfidf, extract_important_words_tfidf
import pyLDAvis.gensim


def train_lda_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    in2word = get_tfidf(documents)["index2word"]

    lda_model = models.ldamodel.LdaModel(corpus=corpus,
                                         num_topics=num_topics,
                                         id2word=in2word,
                                         distributed=False,  # default: False
                                         chunksize=2000,  # default: 2000
                                         passes=1,  # default: 1
                                         update_every=1,  # default: 1
                                         alpha='symmetric',  # default: 'symmetric'
                                         eta=0.3,  # default: None ; ** taking non-default value **
                                         decay=0.5,  # default: 0.5
                                         offset=1.0,  # default: 1.0
                                         eval_every=10,  # default: 10
                                         iterations=50,  # default: 50
                                         gamma_threshold=0.001,  # default: 0.001
                                         minimum_probability=0.01,  # default: 0.01
                                         random_state=None,  # default: None
                                         ns_conf=None,  # default: None
                                         minimum_phi_value=0.01,  # default: 0.01
                                         per_word_topics=False,  # default: False
                                         callbacks=None  # default: None
                                         )

    return lda_model, corpus, in2word


def train_svd_model(data, num_topics):
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = remove_low_high_frequent_words(documents, 0.15, 0.60)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    in2word = get_tfidf(documents)["index2word"]

    svd_model = models.LsiModel(corpus=corpus,
                                num_topics=num_topics,
                                id2word=in2word,
                                chunksize=5,  # default: 20000
                                decay=1.0,  # default: 1.0
                                distributed=False,  # default: False
                                onepass=True,  # default: True
                                power_iters=10,  # default: 2
                                extra_samples=100  # default: 100
                                )

    return svd_model, corpus, in2word


def lda_visualization(data, num_topics):
    model, corpus, dictionary = train_lda_model(data, num_topics)
    print('{} {}'.format('length of the dictionary is: ', len(dictionary)))

    visual = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(visual, 'visual/lda_visual.html')


def plot_word_importance(model):
    plt.figure(figsize=(15, 30))

    for i in range(model.get_topics().shape[0]):  # number of topics
        df = pd.DataFrame(model.show_topic(i), columns=['term', 'prob']).set_index('term')

        plt.subplot(model.get_topics().shape[0]/2, 2, i + 1)  # two plots per row
        plt.title('topic ' + str(i + 1))
        sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
        plt.xlabel('probability')

    plt.show()


def compute_coherence_values_topic_num(data, limit, start=2, step=1):
    """
    Compute u_mass coherence for various number of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    with open(data, 'rb') as file:
        # read the data as binary data stream
        print("... Reading the pre-processed data from local binary file...")
        documents = pickle.load(file)

    documents = extract_important_words_tfidf(documents, 0.60)  # extracting top 60% (TF-IDF) terms per document
    documents = remove_low_high_frequent_words(documents, 0.03, 1.0)

    corpus = get_tfidf(documents)["corpus_tfidf"]
    dictionary = get_tfidf(documents)["index2word"]

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eta=0.3)
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, dictionary=dictionary, corpus=corpus, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def main():
    # ------------- LDA visualization ---------------- #
    lda_visualization('data/case_documents_20000.data', 5)

    # ------------- LDA word importance visualization ---------------- #
    # lda_model = train_lda_model('data/case_documents_5000.data', 10)[0]
    # plot_word_importance(lda_model)

    # ------------- LDA evaluation ---------------- #
    # model_list, coherence_values = compute_coherence_values_topic_num(data='data/case_documents_20000.data',
    #                                                                   limit=15,
    #                                                                   start=2,
    #                                                                   step=1)
    # limit = 15
    # start = 2
    # step = 1
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend("coherence_values", loc='best')
    # plt.show()


if __name__ == '__main__':
    main()
