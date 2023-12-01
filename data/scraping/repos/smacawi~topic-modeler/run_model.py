import pandas as pd
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import json
from preprocessing import clean_tweets
from models import lda, biterm, guided_lda
from util import util
#from models.btm import BitermModel

HASHTAGS = {r'#nlwhiteout', r'#nlweather', r'#newfoundland', r'#nlblizzard2020', r'#nlstm2020', r'#snowmaggedon2020', r'#stmageddon2020', r'#stormageddon2020', r'#newfoundland',
                            r'#snowpocalypse2020', r'#snowmageddon', r'#nlstm', r'#nlwx', r'#nlblizzard', r'#nlwx', 'snowmaggedon2020', 'newfoundland', r'#nlstorm2020', 'snow', 'st'}

if __name__ == '__main__':
    num_topics = [5,10,15,20]
    seed_topic_list = [['traffic', 'road'], ['love','family']]
    crisis_nlp_seeds = [['dead', 'injured', 'hurt', 'people', 'casualties'], ['missing', 'trapped', 'found', 'people'], ['home', 'building', 'roads', 'traffic', 'bridges', 'electricity'],
                        ['donations', 'volunteers', 'shelter', 'supplies'], ['warning', 'advice'], ['thoughts', 'prayers']]
    seed_conf = 0.15
    seed_conf_list = [0.15, 0.30, 0.50, 0.75]

    winter_tweets = pd.read_csv('/home/smacawi/smacawi/repositories/tweet-classifier/nlwx_2020_hashtags_no_rt.csv')
    winter_tweets_cleaned = clean_tweets.preprocess(df=winter_tweets, extra_stopwords = HASHTAGS)
    winter_tweets_cleaned.reset_index(inplace=True, drop=True)
    ready_data = winter_tweets_cleaned['tokenized_text'].values.tolist()
    for d in range(10):
        print(winter_tweets_cleaned['text'].loc[d], ready_data[d])

    # get best LDA model
    #lda.get_best(ready_data, 1, 20, 1)

    '''
    # train best lda by coherence
    best_num = 7
    lda_model = lda.build_model(ready_data, num_topics = best_num, include_vis = True)
    # save top words for each topic to csv
    lda.top_vocab(lda_model, 10)
    '''

    '''
    # trains 5, 10, 15, 20 topics
     for val in num_topics:
         print("Building model with {} topics...".format(val))
         guided_lda_model = guided_lda.build_model(ready_data, num_topics = val, seed_topic_list = seed_topic_list, seed_conf = seed_conf)
         #lda_model = lda.build_model(ready_data, num_topics = val)
         #biterm_model = biterm.build_model(ready_data, num_topics = val)
    '''

    '''
    # guided lda
    for conf in seed_conf_list:
        guided_lda_model = guided_lda.build_model(ready_data, num_topics=9, seed_topic_list=crisis_nlp_seeds,
                                              seed_conf=conf)
    '''

    # self-implemented biterm
    '''
    K = 15
    W = None
    alpha = 0.5
    beta = 0.5
    n_iter = 1000
    save_step = 100
    btm = BitermModel.BitermModel(K, W, alpha, beta, n_iter, save_step)

    btm.run(ready_data, "btm_")
    '''






