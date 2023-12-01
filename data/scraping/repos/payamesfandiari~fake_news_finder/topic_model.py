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

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cleaner


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3,mallet_path='./support/mallet-2.0.8/bin/mallet'):
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
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



if __name__ == '__main__':
    ## Load the data
    data = pd.read_csv('./data/processed/real_or_fake.csv')
    fakes = data.loc[data.label == -1]
    reals = data.loc[data.label == 1]
    # fakes_lemm = cleaner.clean_up_data(fakes.text)
    # print(fakes_lemm[:1])
    #
    # # Create Dictionary
    # id2word = corpora.Dictionary(fakes_lemm)
    #
    # # Create Corpus
    # texts = fakes_lemm
    #
    # # Term Document Frequency
    # corpus = [id2word.doc2bow(text) for text in texts]
    #
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                             id2word=id2word,
    #                                             num_topics=20,
    #                                             random_state=100,
    #                                             update_every=1,
    #                                             chunksize=100,
    #                                             passes=10,
    #                                             alpha='auto',
    #                                             per_word_topics=True)
    # # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    #
    # # Compute Perplexity
    # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    #
    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=fakes_lemm, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)
    #
    #
    #
    # # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    # # mallet_path = './support/mallet-2.0.8/bin/mallet'  # update this path
    # # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
    # #
    # # # Show Topics
    # # pprint(ldamallet.show_topics(formatted=False))
    # #
    # # # Compute Coherence Score
    # # coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=fakes_lemm, dictionary=id2word, coherence='c_v')
    # # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # # print('\nCoherence Score: ', coherence_ldamallet)
    # #
    # # model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=fakes_lemm,
    # #                                                         start=2, limit=40, step=6)
    #
    # vis = pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
    # pyLDAvis.save_html(vis,'fake_lda.html')

    reals_lemm = cleaner.clean_up_data(reals.text.dropna())
    print(reals_lemm[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(reals_lemm)

    # Create Corpus
    texts = reals_lemm

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=reals_lemm, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    # mallet_path = './support/mallet-2.0.8/bin/mallet'  # update this path
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
    #
    # # Show Topics
    # pprint(ldamallet.show_topics(formatted=False))
    #
    # # Compute Coherence Score
    # coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=fakes_lemm, dictionary=id2word, coherence='c_v')
    # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # print('\nCoherence Score: ', coherence_ldamallet)
    #
    # model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=fakes_lemm,
    #                                                         start=2, limit=40, step=6)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'reals_lda.html')
