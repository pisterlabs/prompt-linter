import re

import matplotlib.pyplot as plt
import numpy as np

import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import spacy
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess


def sentences_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts, stop_words):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def clean_words(words, stop_words, trigram_mod=None):

    data_words_nostops = remove_stopwords(words, stop_words)

    # data_words_grams = make_bigrams(data_words_nostops, bigram_mod)

    # if trigram_mod is not None:
    #     data_words_grams = make_trigrams(
    #         data_words_grams, bigram_mod, trigram_mod
    #     )

    # nlp = spacy.load("en", disable=["parser", "ner"]) # VK starting from spacy 3.0 version, "en" is not supported as shortcut anymore
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # VK added this line
    data_lemmatized = lemmatization(
        data_words_nostops, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )

    return data_lemmatized


def compute_coherence_values(
    limit, mallet_path, dictionary, corpus, texts, start=2, step=2
):
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
        model = gensim.models.wrappers.LdaMallet(
            mallet_path,
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_seed=100
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
