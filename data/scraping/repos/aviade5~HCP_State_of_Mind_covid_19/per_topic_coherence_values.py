import sys
import time
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pandas as pd

twitter_model_path = '/gpfs0/rami/users/iliapl/data/output_data/lda_models/70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_20_TOPICS'

print(f'Getting model coherence per topic for {twitter_model_path}', flush=True)

T = time.time()

twitter_corpus = MmCorpus('{}/corpus.mm'.format(twitter_model_path))
twitter_dict = Dictionary.load('{}/dict.id2word'.format(twitter_model_path))
twitter_model = LdaModel.load('{}/lda.model'.format(twitter_model_path))

T = time.time() - T
print('Loaded twitter model, corpus, dictionary in {} seconds'.format(T), flush=True)

T = time.time()

texts = []
for bow in twitter_corpus:
    texts.append([twitter_dict[word_id] for (word_id, _) in bow])

# available coherence measures: u_mass, c_v, c_uci, c_npmi
# explanations can be found here:
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

# coherence_measures = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
coherence_measures = ['c_v']

for coherence_measure in coherence_measures:
    cT = time.time()
    print(f'Calculating coherence measure {coherence_measure}...', flush=True)
    coherence_model = CoherenceModel(model=twitter_model, texts=texts, corpus=twitter_corpus, dictionary=twitter_dict,
                                     coherence=coherence_measure)
    coherence_per_topic = coherence_model.get_coherence_per_topic()
    pd.DataFrame({'topic_coherence': coherence_per_topic}).to_csv('{}/coherence_per_topic.csv'.format(twitter_model_path))


print(f'Computed coherence scores in {time.time() - T} seconds', flush=True)