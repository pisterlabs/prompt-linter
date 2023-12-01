from json import dump
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from gensim.models import HdpModel, LdaModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_documents, remove_stopwords
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import get_pruned_data, get_summaries, get_ext_summaries


to_remove_summaries = ['participant was asked', 'then participant said', 'nan']


def get_topic_model(conversation_processed, model_type, num_topics=None):
    docs = conversation_processed
    dct = Dictionary(docs)
    corpus = [dct.doc2bow(row) for row in docs]
    if model_type == 'hdp':
        model = HdpModel(corpus, id2word=dct)
    elif model_type == 'lda':
        model = LdaModel(corpus, id2word=dct, num_topics=num_topics)
    return model, dct, corpus


def KL(original, approximation):
    a = np.asarray(original, dtype=np.float)
    b = np.asarray(approximation, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def JS(a, b):
    return jensenshannon(a, b)


def my_preprocess(docs):
    temp = [remove_stopwords(s) for s in docs]
    return [d.split(' ') for d in temp]
    # return preprocess_documents(docs)


def get_dist(doc_bow, topic_model, model_type, num_topics=None):
    if model_type == 'hdp':
        num_topics = len(topic_model.hdp_to_lda()[0])
    doc_topics = topic_model[doc_bow]
    # print([tup[0] for tup in doc_topics])
    doc_dist = np.zeros(num_topics)
    for idx, value in doc_topics:
        doc_dist[idx] = value
    current_sum = doc_dist.sum()
    zeros = np.argwhere(doc_dist == 0)
    if zeros.size != 0:
        to_fill = (1-current_sum) / len(zeros)
        doc_dist[zeros] = to_fill
    assert 1 - doc_dist.sum() < 0.0001
    return doc_dist


def preprocess_conversation(conversation):
    docs = []
    for _row_idx, row in conversation.iterrows():
        utterance = str(row['value'])
        if row['speaker'] == 'ellie':
            docs.append(utterance)
        elif row['speaker'] == 'participant':
            docs[-1] += ' ' + utterance
    if '' in docs:
        print('FOUND EMPTY STR')
    # print(docs)
    # docs = [d.split(' ') for d in docs]
    docs = my_preprocess(docs)
    return docs


def preprocess_summary(summary, dct):
    result = summary
    for to_remove in to_remove_summaries:
        result = result.replace(to_remove, '')
    result = my_preprocess(result.split('.'))
    result = [x for x in result if x]  # remove empty lists
    return result


def get_dct_model_dists(data, num_topics, model_type):
    dcts, models, dists, corpuses = {}, {}, {}, {}
    for pid, conversation in data.items():
        conversation_processed = preprocess_conversation(conversation)
        # topicModel, dct = get_hdp_model(conversation_processed, model_type='hdp)
        topicModel, dct, corpus = get_topic_model(
            conversation_processed, model_type='lda', num_topics=num_topics)
        dcts[pid], models[pid], corpuses[pid] = dct, topicModel, corpus
        total_conv = pd.Series(conversation_processed).sum()
        conv_bow = dct.doc2bow(total_conv)
        dists[pid] = get_dist(conv_bow, topicModel,
                              model_type=model_type, num_topics=num_topics)
    return dcts, models, dists, corpuses


# SEM_AUG, AUG, EXT, NORMAL, OVER_EXT
def topic_evaluation(algo, topic_models, dcts, dists, model_type, num_topics):
    all_num_topics = {}
    lim = 7
    extractive_algo = 'SumBasic_unpruned'
    if algo == 'ES':
        summaries = get_ext_summaries(extractive_algo)
    elif algo == 'KIAS':
        summaries = get_summaries(
            augmented_lm=True, sem=True, lim=lim, over_ext=False)
    elif algo == 'OKIAS':
        summaries = get_summaries(
            augmented_lm=True, sem=False, lim=lim, over_ext=False)
    elif algo == 'AS':
        summaries = get_summaries(
            augmented_lm=False, sem=False, lim=lim, over_ext=False)
    elif algo == 'AOES':
        summaries = get_summaries(
            augmented_lm=False, sem=False, lim=lim, over_ext=True)
    rows = []
    for pid, topicModel in topic_models.items():
        dct = dcts[pid]
        summary_processed = preprocess_summary(summaries[pid], dct)

        total_sum = pd.Series(summary_processed).sum()
        try:
            summary_bow = dct.doc2bow(total_sum)
            summary_dist = get_dist(
                summary_bow, topicModel, model_type=model_type, num_topics=num_topics)
        except:
            print(f'exception at pid={pid}, algo={algo}, n={num_topics}')
            summary_dist = np.ones(num_topics) / num_topics
        row = {}
        row['KL'] = KL(original=dists[pid], approximation=summary_dist)
        row['JS'] = JS(dists[pid], summary_dist)
        row['num_topics'] = num_topics
        row['pid'] = pid
        rows.append(row)
    return rows
    '''
        metrics['KL'] = KL(original=dists[pid], approximation=summary_dist)
        metrics['JS'] = JS(dists[pid], summary_dist)
        try:
            with open(f'{dest}/{pid}.json', 'w') as fp:
                dump(metrics, fp)
        except FileNotFoundError:
            os.mkdir(dest)
            with open(f'{dest}/{pid}.json', 'w') as fp:
                dump(metrics, fp)
    '''

    # ax = sns.distplot(all_num_topics.values(), kde=False)
    # plt.show()


def get_coherence_scores(topic_models, corpuses, dcts):
    coherences = {}
    for pid in topic_models.keys():
        topicModel, corpus, dct = topic_models[pid], corpuses[pid], dcts[pid]
        cm = CoherenceModel(model=topicModel, corpus=corpus, coherence='u_mass', dictionary=dct)
        coherences[pid] = cm.get_coherence()
    return coherences


def main():
    algos = ('KIAS', 'ES', 'AS', 'AOES')
    model_type = 'lda'
    data, _labels = get_pruned_data()
    dataframes = {}
    for algo in algos:
        dataframes[algo] = pd.DataFrame(columns=['pid', 'num_topics', 'KL', 'JS'])
    for num_topics in range(2, 9):
        dcts, lda_models, dists, corpuses = get_dct_model_dists(
            data, num_topics, model_type)
        
        # COHERENCES
        coherences = get_coherence_scores(lda_models, corpuses, dcts)
        values = np.array(list(coherences.values())).flatten()
        print(f'n = {num_topics} MED, STD COHERENCE = {np.median(values)}, {values.std()}')
        # continue
        for algo in algos:
            rows = topic_evaluation(algo, lda_models, dcts,
                             dists, model_type, num_topics)
            dataframes[algo] = dataframes[algo].append(rows)
        print(f'DONE WITH n={num_topics}')
    
    for algo, df in dataframes.items():
        df.to_csv(f'../evaluation/{algo}.tsv', sep='\t', index=False)
    


if __name__ == '__main__':
    main()
