"""
LDA script

"""
import gensim#, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import STOPWORDS

def run_lda(corpus, id2word):
# Build LDA model
	lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=8)
	#LdaMulticore uses multiple workers 
#	lda_model = gensim.models.LdaMulticore(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=8, 
#                                             random_state=100,
#                                             chunksize=10,
#                                             passes=10,
#                                             workers=3,
#                                             alpha='symmetric',
#                                             iterations=100,
#                                             per_word_topics=False)
	return lda_model