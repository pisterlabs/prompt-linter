import pandas as pd
from gensim import corpora
import sys

from gensim.models import CoherenceModel
sys.path.append("..")
import numpy as np
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim as gensimvis

if __name__ == '__main__':
    darklyrics = pd.read_csv('../darklyrics-proc-tokens-single.csv',
                             converters={'tokens': lambda x: x.strip("[]").replace("'", "").split(", ")})

    documents = darklyrics['tokens']
    labels = darklyrics['genre']

    id2word = corpora.Dictionary(documents)
    corpus = documents.apply(lambda x: id2word.doc2bow(x))

    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=9,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=documents.as_matrix(), dictionary=id2word,
                                         coherence='c_v', processes=1)
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    vis_data = gensimvis.prepare(lda_model, corpus, id2word)