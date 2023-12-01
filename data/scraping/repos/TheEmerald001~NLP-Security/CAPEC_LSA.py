import pickle

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LsiModel
# Plotting tools
import pyLDAvis.gensim_models  # don't skip this


for x in range(1, 545):
    capec_corpus = corpora.MmCorpus("CAPEC/CORPUSES/capec_corpus_" + str(x))
    capec_dictionary = corpora.Dictionary.load("CAPEC/DICTIONARIES/capec_dictionary_" + str(x))
    capec_text = pickle.load(open("CAPEC/AP TEXTS/capec_entry_" + str(x), 'rb'))
    lsa_model = gensim.models.lsimodel.LsiModel(corpus=capec_corpus,
                                                id2word=capec_dictionary,
                                                chunksize=100,
                                                num_topics=4,
                                                )
    # Save lsa model to disk
    lsa_model.save("CAPEC/LSA/capec_lsa_" + str(x))








