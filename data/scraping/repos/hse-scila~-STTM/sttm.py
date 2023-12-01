import numpy as np
import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import CoherenceModel
from gensim.models.wrappers.dtmmodel import DtmModel
from scipy.stats import pearsonr

def get_cross_validation_inds(start_date, train_date, end_date, offset, mode='expanding'):

    start_date = pd.to_datetime(start_date)
    train_date = pd.to_datetime(train_date)    
    end_date = pd.to_datetime(end_date)
        
    train_dates = []
    test_dates = []
        
    while train_date + pd.offsets.Day() < end_date:

        train_dates.append([start_date, train_date])
        test_dates.append([train_date + pd.offsets.Day(), train_date + offset])  
                
        if mode == 'rolling':
            start_date = start_date + offset + pd.offsets.Day()
        
        train_date += offset
               
    return train_dates, test_dates
 
# train/test dates
# val_train_dates, val_test_dates = get_cross_validation_inds('2013-01-01', '2014-12-31', '2021-01-01', 
#                                                             pd.offsets.YearEnd(), mode='expanding')

def get_doc_topics_matrix(bow, topic_model):

    doc_topics_matrix = np.zeros(shape=(len(bow), topic_model.num_topics))
    for i in tqdm_notebook(range(doc_topics_matrix.shape[0])):
        doc_topics_matrix[i] = np.array(topic_model.get_document_topics(bow[i], minimum_probability=0))[:, 1]
        
    return doc_topics_matrix

def get_topic_stream(doc_topics_matrix, time_inds):
    
    topic_stream = np.zeros(shape=(len(time_inds), doc_topics_matrix.shape[1]))
    
    for i, inds in enumerate(time_inds):
        
        if len(inds):
            tc = np.zeros(doc_topics_matrix.shape[1])
            tc = doc_topics_matrix[inds].sum(axis=0)
        else:
             tc = np.array([np.nan for _ in range(doc_topics_matrix.shape[1])])
        
        topic_stream[i] = tc
        
    return topic_stream

def get_d_matrix(bow, time_inds, dictionary):

    d_matrix = np.zeros(shape=(len(time_inds), len(dictionary)))
    for i, inds in enumerate(time_inds):
        if len(inds):
            for ind in inds:
                doc = bow[ind]
                if doc:
                    doc = np.array(bow[ind])
                else:
                    continue
                d_matrix[i][doc[:, 0]] += doc[:, 1]
        else:
            d_matrix[i, :] = np.array([np.nan for _ in range(len(dictionary))])

    return d_matrix

def get_word_stream(d_matrix, topic_model, topic_id, prob_mass=0.3):
    
    assert topic_id < topic_model.num_topics
    
    topic_word_matrix = np.array(topic_model.get_topic_terms(topic_id, topn=topic_model.num_terms))
    non_zeroes_word = np.where(d_matrix.sum(axis=0) != 0)[0]
    topic_word_matrix = topic_word_matrix[non_zeroes_word]

    for cutoff in range(1, topic_model.num_terms + 1):
        if (topic_word_matrix[:cutoff][:, 1].sum() > prob_mass):
            break
            
    cutoff -= 1

    top_words = np.array(topic_word_matrix[:cutoff][:, 0], dtype='int64')
    word_stream = d_matrix[:, top_words]
    
    return word_stream, top_words

def get_words_significance(timeSeries, word_stream):
    
    assert timeSeries.shape == (word_stream.shape[0],)
        
    X = word_stream.copy()
    target = timeSeries.values
    
    nan_inds = np.unique(np.argwhere(np.isnan(target))[:, 0])
    nan_inds = np.append(nan_inds, np.unique(np.argwhere(np.isnan(X))[:, 0]))
    target = np.delete(target, nan_inds)
    X = np.delete(X, nan_inds, axis=0)
    
    corr_p_matrix = np.zeros((X.shape[1], 2))
    for i in range(X.shape[1]):
        corr_value, p_value = pearsonr(target, X[:, i])
        corr_p_matrix[i][0] = corr_value
        corr_p_matrix[i][1] = p_value
        
    return corr_p_matrix

def get_topics_tonality(bow, dictionary, time_inds, TM_model, timeSeries, params, verbose=False):
    
    assert len(time_inds) == len(timeSeries)
    
    prob_masses = params.get('prob_masses')
    p_value = params.get('p_value')
    
    d_matrix = get_d_matrix(bow, time_inds, dictionary)
    tonality = np.zeros(TM_model.num_topics)
    
    words = set()
    significant_words = {}
    
    for topic in range(TM_model.num_topics):
        word_stream, top_words = get_word_stream(d_matrix, TM_model, topic, prob_masses[topic])
        
        mask = np.array([dictionary[i] not in words for i in top_words])
        masked_top_words = top_words[mask]
        masked_word_stream = word_stream[:, mask]
                
        if masked_top_words.size != 0:
            corr_p_matrix = get_words_significance(timeSeries, masked_word_stream)
            significant_ind = np.where(corr_p_matrix[:, 1] < p_value)[0]
            
            words.update([dictionary[i] for i in masked_top_words])
            significant_words.update({dictionary[key]: value for key, value 
                                      in zip(masked_top_words[significant_ind],
                                             corr_p_matrix[significant_ind][:, 0])})
            
        signif_values = np.array([significant_words.get(dictionary[i], np.nan) for i in top_words])
        signif_mask = ~np.isnan(signif_values)
        signif_top_words = top_words[signif_mask]
        signif_values = signif_values[signif_mask]
                
        if signif_top_words.size == 0:
            if verbose:
                print('There are not significant word-correlation. Topic: {}'.format(topic))
            pProb = 0
            nProb = 0
            continue
            
        probs = np.array(TM_model.get_topic_terms(topic, topn=len(top_words)))[:, 1][signif_mask]
        contribution = signif_values * probs

        pProb = np.sum(np.abs(contribution[contribution >= 0])) / np.sum(np.abs(contribution))
        nProb = np.sum(np.abs(contribution[contribution < 0])) / np.sum(np.abs(contribution))
            
        tonality[topic] = pProb - nProb 
            
    return tonality, significant_words, words


def get_dtm_word_stream(d_matrix, dtm_model, topic_id, time_id, dictionary, prob_mass=0.3):
    
    assert topic_id < dtm_model.num_topics
    assert time_id < len(dtm_model.time_slices)
    assert dtm_model.num_terms == len(dictionary)
     
    dtm_time_words_df = pd.DataFrame(dtm_model.show_topic(topic_id, time_id, 
                                                          topn=dtm_model.num_terms), columns=['prob', 'word'])
    dtm_time_words_df['word_id'] = dtm_time_words_df['word'].apply(lambda x: dictionary.token2id[x])
    topic_time_word_matrix = dtm_time_words_df[['word_id', 'prob']].values
    
    for cutoff in range(1, dtm_model.num_terms + 1):
        if (topic_time_word_matrix[:cutoff][:, 1].sum() > prob_mass):
            break
    cutoff -= 1
    
    top_words = np.array(topic_time_word_matrix[:cutoff][:, 0], dtype='int64')
    word_stream = d_matrix[:, top_words]
    
    return word_stream, top_words

def get_dtm_topics_tonality(share, dtm_model, time_slices, train_dates, params, verbose=False):
        
    bow = share['bow']
    dictionary = share['dictionary']
    
    prob_masses = params.get('prob_masses')
    p_value = params.get('p_value')
    
    train_time_inds = share['news_df'].resample(rtime)\
                .ind.unique()[train_dates[0]:train_dates[-1]].apply(lambda x: list(x))\
                    .reindex(share['ts'][train_dates[0]:train_dates[-1]].index)
    for row in train_time_inds.loc[train_time_inds.isnull()].index:
        train_time_inds.at[row] = []
    train_time_inds = train_time_inds.values.tolist()
    
    d_matrix = get_d_matrix(bow, train_time_inds, dictionary)
    
    words = set()
    significant_words = {}
    dtm_topic_dict = {}
    
    dtm_tonality_matrix = np.zeros(shape=(dtm.num_topics, len(share['inds'])))
    
    for topic_id in tqdm_notebook(range(dtm.num_topics)):
        
        time_top_words = {}
        time_signif_words = {}
        
        for time_ind in range(len(share['inds'])):
            
            try:
                # if there is no news data in the week, then the index is nan
                date = share['news_df'].iloc[:share['inds'][time_ind][-1] + 1].date[-1]
            except:
                dtm_tonality_matrix[topic_id][time_ind] = np.nan
                continue
                
            time_id = np.where(np.array(time_slices) >= date)[0][0]
                                                                    
            word_stream, top_words = get_dtm_word_stream(d_matrix, dtm, topic_id, time_id, dictionary, 
                                                         prob_mass=prob_masses[topic_id])
            
            time_top_words[time_ind] = {dictionary[word_id] for word_id in top_words}
            
            mask = np.array([w not in words for w in time_top_words[time_ind]])
            masked_top_words = top_words[mask]
            masked_word_stream = word_stream[:, mask]
            
            if masked_top_words.size != 0:
                corr_p_matrix = get_words_significance(share['ts'][train_dates[0]:train_dates[-1]], 
                                                       masked_word_stream)
                significant_ind = np.where(corr_p_matrix[:, 1] < p_value)[0]
                words.update([dictionary[i] for i in masked_top_words])

                significant_words.update({dictionary[key]: value for key, value 
                                  in zip(masked_top_words[significant_ind],
                                         corr_p_matrix[significant_ind][:, 0])})
                
            signif_values = np.array([significant_words.get(dictionary[i], np.nan) for i in top_words])
            signif_mask = ~np.isnan(signif_values)
            signif_top_words = top_words[signif_mask]
            signif_values = signif_values[signif_mask] 

            time_signif_words[time_ind] = {dictionary[key]: value for key, value in zip(signif_top_words,
                                                                                        signif_values)}

            if signif_top_words.size == 0:
                if verbose:
                    print('There are not significant word-correlation. Topic: {}, Time: {}'.format(topic_id, time_id))
                pProb = 0
                nProb = 0
                continue

            probs = np.array(dtm.show_topic(topic_id, time_id, 
                                            topn=len(top_words)))[:, 0][signif_mask].astype('float64')
            contribution = signif_values * probs

            pProb = np.sum(np.abs(contribution[contribution >= 0])) / np.sum(np.abs(contribution))
            nProb = np.sum(np.abs(contribution[contribution < 0])) / np.sum(np.abs(contribution))
            
            dtm_tonality_matrix[topic_id][time_ind] = pProb - nProb 
        dtm_topic_dict[topic_id] = {'time_top_words': time_top_words, 'time_signif_words': time_signif_words}
        
    return dtm_tonality_matrix, dtm_topic_dict, words, significant_words


################################################################################################################################
# if mallet_flg:
#     mallet_path = './mallet-2.0.8/bin/mallet'
#     lda_mallet = LdaMallet(mallet_path, corpus=corpus_kom, 
#                            num_topics=num_topics, id2word=dictionary_kom, 
#                            random_seed=11)
#     tm_model_kom = malletmodel2ldamodel(lda_mallet)
# else:
#     tm_model_kom = LdaModel(corpus_kom, num_topics=num_topics, 
#                         random_state=11, id2word=dictionary_kom)

# dtm = DtmModel.load('../dtm_models/dtm_20topics_kommersant')
# inds = news_kommersant.resample('1M').ind.unique().apply(lambda x: list(x)).values.tolist()
# time_seq = [len(ind) for ind in inds]

# cum_slice = 0

# time_slices = []
# for tslice in dtm.time_slices:
#     cum_slice += tslice
#     time_slices.append(news_kommersant.iloc[:cum_slice].date[-1])
#################################################################################################################################


#################################################################################################################################

# sttm (lda)

# params = {'prob_masses':np.array([0.3 for _ in range(TM_model.num_topics)]), 'p_value':0.05}

# tonality, significant_words, words  = get_topics_tonality(share['bow'], 
#                                                           share['dictionary'],
#                                                           train_time_inds,
#                                                           share['tm_model'], 
#                                                           share['ts'][train_dates[0]:train_dates[1]], 
#                                                           params)

# sttm_lda_df = pd.DataFrame((share['topic_stream'] * tonality).sum(axis=1), 
#                             index=share['ts'].index).dropna()



# sttm (dtm)

# params = {'prob_masses':np.array([0.3 for _ in range(TM_model.num_topics)]), 'p_value':0.05}

# topic_stream_dtm = get_topic_stream(dtm.gamma_, shares[shares_name]['inds'])

# dtm_tonality_matrix, dtm_topic_dict, words, significant_words =\
#     get_dtm_topics_tonality(shares[shares_name], dtm, time_slices, train_dates, params)

# sttm_dtm_df = pd.DataFrame((dtm_tonality_matrix.T * topic_stream_dtm).sum(axis=1), 
#                        index=shares[shares_name]['ts'].index)

#################################################################################################################################

#################################################################################################################################
# coherence score

# limit = 50
# start=2
# step=3
    
# coherence_values = []
# model_list = []

# for num_topics in tqdm_notebook(range(start, limit, step)):
    
#     tm_model_kom = LdaModel(corpus_kom, num_topics=num_topics, 
#                         random_state=11, id2word=dictionary_kom)
#     model_list.append(tm_model_kom)
#     coherencemodel = CoherenceModel(model=tm_model_kom, texts=primary_corpus_kom, 
#                                     dictionary=dictionary_kom, coherence='c_v')
#     coherence_values.append(coherencemodel.get_coherence())