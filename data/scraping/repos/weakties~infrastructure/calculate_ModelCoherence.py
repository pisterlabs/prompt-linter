from LdaRecsys import LdaRecsys
from gensim.models import CoherenceModel
import numpy as np

import pickle

def coherence_score(model):
    topics=[]

    for i in range(model.lda.num_topics):
        terms = []
        for n in model.lda.show_topic(i):
            terms.append(n[0])
    topics.append(terms)
    
    model_corpus = [corp for corp in list(model.corp_bow.values())]
    print(len(model_corpus))
    model_dictionary = model.corp_dict
    alltexts = [user_text for user_text in list(model.user_corp.values())]
    print(len(alltexts))

    cm_umass = CoherenceModel(topics=topics, corpus=model_corpus, dictionary=model_dictionary, coherence='u_mass')
    cm_cv = CoherenceModel(topics=topics, texts=alltexts, dictionary=model_dictionary, coherence='c_v')
    cm_cuci = CoherenceModel(topics=topics, texts=alltexts, dictionary=model_dictionary, coherence='c_uci')
    cm_cnpmi = CoherenceModel(topics=topics, texts=alltexts, dictionary=model_dictionary, coherence='c_npmi')

    return (model.lda.num_topics,cm_umass.get_coherence(), cm_cv.get_coherence(), cm_cuci.get_coherence(), cm_cnpmi.get_coherence())

lda = LdaRecsys()

lda.buildCorpDict()

lda.buildCorpBow()

iter_result = []
for n in [50]:
    
    for i in np.arange(5,150,1):

        coherence = []
        lda.trainLDA(i,n)
    
        result = coherence_score(lda)

        print('iterat: {}  scorre: {}'.format(n,result))
        coherence.append(result)

        iter_result.append((n, coherence))



with open('coherence_score_50_iteration.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(iter_result, f)

