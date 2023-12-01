import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
import pickle
import logging
#import nltk
#nltk.download('wordnet')
from multiprocessing import Process, freeze_support
import numpy as np
from gensim.topic_coherence import direct_confirmation_measure  #https://github.com/RaRe-Technologies/gensim/issues/3040
from gensim_fix import custom_log_ratio_measure



#Some code inspired from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 & 
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

if __name__ == '__main__':
    freeze_support()
    # SETTINGS FOR MODEL
    RANDOM_SEED = 7245
    chunk_size = 2000
    passes = 5
    num_topics=21
    dic_file = "models/bi_trained_lda_dictionary.sav"
    corp_file = "models/bi_trained_lda_corpus_tfidf.sav"
    model_file = "models/bi_trained_lda.sav"
    text_file = "models/trained_lda_texts.sav"
    texts = pickle.load(open(text_file, 'rb')) 
    #for gensim to output some progress information while it's training
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print ("Loading the dic, corpus and model")
    dictionary = pickle.load(open(dic_file, 'rb'))
    corpus = pickle.load(open(corp_file, 'rb'))  
    print ("Training the model")

    #Lda model with settings
    #LDA = gensim.models.ldamodel.LdaModel
    #ldamodel = LDA(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=RANDOM_SEED)
    LDA = gensim.models.LdaMulticore
    ldamodel = LDA(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=RANDOM_SEED, chunksize=chunk_size, dtype=np.float64,  alpha="symmetric", eta=1) 
    #Save the LDA Model
    pickle.dump(ldamodel, open(model_file, 'wb'))

#from pprint import pprint
# Print the Keyword in the 10 topics
# pprint(ldamodel.print_topics())
    direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
    print ("Coherence with 21 topics, alpha=symmetric and beta=1 : {}".format(coherence_model_lda.get_coherence()))

    