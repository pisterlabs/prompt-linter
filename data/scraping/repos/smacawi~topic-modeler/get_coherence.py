import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import json
from preprocessing import clean_tweets
from models import lda, biterm, guided_lda
from util import util
from operator import itemgetter

HASHTAGS = {r'#nlwhiteout', r'#nlweather', r'#newfoundland', r'#nlblizzard2020', r'#nlstm2020', r'#snowmaggedon2020', r'#stmageddon2020', r'#stormageddon2020', r'#newfoundland',
                            r'#snowpocalypse2020', r'#snowmageddon', r'#nlstm', r'#nlwx', r'#nlblizzard', r'#nlwx', 'snowmaggedon2020', 'newfoundland', r'#nlstorm2020', 'snow', 'st'}

if __name__ == "__main__":
    topics = [5, 9, 10, 15, 20]
    #preprocessing data
    winter_tweets = pd.read_csv('/home/smacawi/smacawi/repositories/tweet-classifier/nlwx_2020_hashtags_no_rt.csv')
    winter_tweets_cleaned = clean_tweets.preprocess(df=winter_tweets, extra_stopwords = HASHTAGS)
    winter_tweets_cleaned.reset_index(inplace=True, drop=True)
    ready_data = winter_tweets_cleaned['tokenized_text'].values.tolist()

    #df to store coherence scores
    df = pd.DataFrame(columns=['model', 'u_mass', 'c_v', 'c_uci', 'c_npmi'])
    id2word = corpora.Dictionary(ready_data)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in ready_data]
    #counter for df
    i=0

    for val in topics:
        if val != 9:
            continue
        print("Building model with {} topics...".format(val))
        #coherence model for LDA
        lda_model = lda.build_model(ready_data, num_topics = val)
        u_mass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass').get_coherence()
        c_v = CoherenceModel(model=lda_model, texts=ready_data, dictionary=id2word, coherence='c_v').get_coherence()
        c_uci = CoherenceModel(model=lda_model, texts=ready_data, dictionary=id2word, coherence='c_uci').get_coherence()
        c_npmi = CoherenceModel(model=lda_model, texts=ready_data, dictionary=id2word, coherence='c_npmi').get_coherence()
        lda_coh = [u_mass, c_v, c_uci, c_npmi]
        df.loc[i] = ['lda_{}'.format(val)] + lda_coh
        i += 1


        #coherence model for BTM
        btm, biterm_model = biterm.build_model(ready_data, num_topics=val)
        #save top words
        top_words = biterm.top_vocab('./btm_summary_{}.npy'.format(val))
        u_mass_b = CoherenceModel(topics=top_words, corpus=corpus, dictionary=id2word, coherence='u_mass').get_coherence()
        c_v_b = CoherenceModel(topics=top_words, texts=ready_data, dictionary=id2word, coherence='c_v').get_coherence()
        c_uci_b = CoherenceModel(topics=top_words, texts=ready_data, dictionary=id2word, coherence='c_uci').get_coherence()
        c_npmi_b = CoherenceModel(topics=top_words, texts=ready_data, dictionary=id2word,
                                coherence='c_npmi').get_coherence()
        btm_coh = [u_mass_b, c_v_b, c_uci_b, c_npmi_b]
        df.loc[i] = ['btm_{}'.format(val)] + btm_coh
        i += 1


        #crisisnlp evaluation
        if val == 9:
            #get labels for text
            lda_labels = pd.DataFrame(columns=['text', 'label'])
            lda_top_words = pd.DataFrame()

            #retrieve top words for each topic
            for i in range(lda_model.num_topics):
                lda_top_words['topic_{}'.format(i)] = pd.Series([i[0] for i in lda_model.show_topic(i, topn=10)])
            lda_top_words.to_csv('./results/lda/lda_top_words_9.csv')

            #retrieve lda topic label for each document
            for d in range(len(ready_data)):
                doc_bow = lda_model.id2word.doc2bow(ready_data[d])
                conf = lda_model[doc_bow][0]
                lda_labels.loc[d] = [winter_tweets_cleaned['text'].loc[d]] + [max(conf,key=itemgetter(1))[0]]
            lda_labels.to_csv('./results/lda_labels.csv')

            #retrieve btm topic label for each document
            btm_labels = pd.DataFrame(columns=['text', 'label'])
            for d in range(len(ready_data)):
                btm_labels.loc[d] = [winter_tweets_cleaned['text'].loc[d]]+[biterm_model[d].argmax()]
            btm_labels.to_csv('./results/btm_labels.csv')


    #save dataframe
    df.to_csv('./results/coherence_baselines.csv')