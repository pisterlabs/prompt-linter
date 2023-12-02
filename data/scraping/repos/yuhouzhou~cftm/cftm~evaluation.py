from gensim.models.coherencemodel import CoherenceModel
import pickle

lda = pickle.load(open('../output/lda_model', 'rb'))

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda['model'], texts= lda['texts'], corpus=lda['corpus'], dictionary=lda['dictionary'],
                                     coherence='u_mass')
lda['coherence'] = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', lda['coherence'])
