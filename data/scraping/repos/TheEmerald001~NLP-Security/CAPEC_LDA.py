# Gensim
import gensim
import pickle
import gensim.corpora as corpora
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis.gensim_models  # don't skip this

for x in range(1, 545):
    capec_corpus = corpora.MmCorpus("CAPEC/CORPUSES/capec_corpus_" + str(x))
    capec_dictionary = corpora.Dictionary.load("CAPEC/DICTIONARIES/capec_dictionary_" + str(x))
    capec_text = pickle.load(open("CAPEC/AP TEXTS/capec_entry_" + str(x), 'rb'))
    lda_model = gensim.models.ldamodel.LdaModel(corpus=capec_corpus, id2word=capec_dictionary,num_topics=4, random_state=100, update_every=1, chunksize=100,passes=10,alpha='auto',per_word_topics=True)
    lda_model.save("CAPEC/LDA/capec_lda_"+str(x))




