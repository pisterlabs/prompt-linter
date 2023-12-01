import re
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from tqdm import tqdm


# Meta optimalization - selection of the best number of topics for a model:
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, mallet_lda = False):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Trying LDA model with",num_topics,"topics.")
        mallet_path = '../mallet-2.0.8/bin/mallet'  # update this path
        if mallet_lda:
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        else:
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def LDA_best_number_of_topics(id2word, corpus, texts, topics_start, topics_end, topics_step, mallet_lda, plot_name):

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts,
                            start=topics_start, limit=topics_end, step=topics_step, mallet_lda=mallet_lda)

    # Show graph
    x = range(topics_start, topics_end, topics_step)
    print("x (Num Topics)",x)
    print("coherence_values",coherence_values)

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(plot_name)
    #plt.show()
    plt.close()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Statistics:

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    dominant_topics_as_arr = []

    # Get main topic in each document
    for i, document in enumerate(ldamodel[corpus]):
        belonging_to_topic = document[0]
        belonging_to_topic = sorted(belonging_to_topic, key=lambda x: (x[1]), reverse=True)
        topic_num, prop_topic = belonging_to_topic[0]
        dominant_topics_as_arr.append(topic_num)
        wp = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                                               ignore_index=True)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df, dominant_topics_as_arr

# Helper functions:

def sentences_to_words(sentences):
    for sentence in sentences:
        # remove accent, remove too short and too long words
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(bigram_model, texts):
    return [bigram_model[doc] for doc in texts]


def make_trigrams(trigram_model, bigram_model, texts):
    return [trigram_model[bigram_model[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
