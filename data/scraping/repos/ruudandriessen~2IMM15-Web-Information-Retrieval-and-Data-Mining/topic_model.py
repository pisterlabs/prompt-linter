from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models import LdaModel

import os
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
TEMP_FOLDER = os.path.join(os.path.sep, os.getcwd(), 'temp/')


# docid_list_String = key: docID, value: list(NOT set) of all words appearing in the doc in the order
#  of appearance
def save_gensim_dict(docID_list_strings):
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    words_per_docs = [coll_words for docId, coll_words in (filter_tokens(docID_list_strings)).items()]
    dictionary = corpora.Dictionary(words_per_docs)
    dictionary.filter_extremes(no_below=500, no_above=0.5)
    # store the dictionary, for future reference
    dictionary.save(os.path.join(TEMP_FOLDER, 'data_gensim.dict'))
    save_gensim_corpus(words_per_docs, dictionary)


def save_gensim_corpus(words_per_docs, gensim_dict):
    corpus = [gensim_dict.doc2bow(words) for words in words_per_docs]
    # store to disk, for later use
    try:
        corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.mm'), corpus)
        print("Gensim preparation of docs is successful: check temp folder")

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


def get_gensim_dict():
    if os.path.exists(os.path.join(TEMP_FOLDER, 'data_gensim.dict')):
        try:
            dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'data_gensim.dict'))
            print("Used files generated from the last indexing")
            return dictionary

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def get_corpus():
    if os.path.exists(os.path.join(TEMP_FOLDER, 'corpus.mm')):
        try:
            corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'corpus.mm'))
            print("Loading corpus.. ")
            return corpus
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def do_lda_modelling(corpus, dictionary, topcnr= 30):
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topcnr,
                                   alpha='auto', eta='auto', update_every=0, chunksize=3500, passes=50)
    save_lda_model(lda)
    return lda


# filtering all the tokens, with 1 character/turns out there are many of them
# does not allow good topic modeling
def filter_tokens(docs_tokens):
    filtr_docs_tokens = dict()
    lemmatizer = WordNetLemmatizer()
    for docId, tokens in docs_tokens.items():
        filtr_docs_tokens[docId] = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 2
                                    and (not token.isdigit())]
    return filtr_docs_tokens


def save_lda_model(lda_model):
    if os.path.exists(os.path.join(TEMP_FOLDER)):
        try:
            lda_model = lda_model.save(os.path.join(TEMP_FOLDER, 'lda_model.lda'))
            print("Saving lda model.. ")
            return lda_model
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def load_ldamodel():
    if os.path.exists(os.path.join(TEMP_FOLDER, 'lda_model.lda')):
        try:
            lda_model = models.LdaModel.load(os.path.join(TEMP_FOLDER, 'lda_model.lda'))
            print("Loading lda model.. ")
            return lda_model
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def label_doc(doc_lda_result):
    max_probability = 0
    tid_max_prob = -1 #topic id of topic with highest probability

    for i in range(0, len(doc_lda_result)):
        if doc_lda_result[i][1] > max_probability:
            # the second value from the tuple is probability
            max_probability = doc_lda_result[i][1]
            # the first value from the tuple is topicId
            tid_max_prob = doc_lda_result[i][0]
    return tid_max_prob


def evaluate_graph(corpus, dictionary, limit=30, plot_graph=False):
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        c_v.append(cm.get_coherence())

        # Show graph
        if plot_graph:
            import matplotlib.pyplot as plt
            x = range(1, limit)
            plt.plot(x, c_v)
            plt.xlabel("num_topics")
            plt.ylabel("Coherence score")
            plt.legend(("c_v"), loc='best')
            plt.show()
    return lm_list


def map_topic_ids(topics):
    topics_mapping = {
        0: 'Pattern recognition optimization',
        1: 'Graph theory',
        2: 'Neural net design',
        3: 'Image recognition',
        4: 'Stochastic process',
        5: 'Neural net application',
        6: 'Statistical learning',
        7: 'Mathematical model',
    }
    result = []
    for topic_id in topics:
        result.append({'name': topics_mapping[topic_id], 'id': topic_id})
    return result
