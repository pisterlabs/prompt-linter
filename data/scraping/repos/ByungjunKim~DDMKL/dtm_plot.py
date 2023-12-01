import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
tqdm.pandas()
import pickle
import os

import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm

if os.name=='nt':
    # Windows
    plt.rc('font', family='Malgun Gothic')
elif os.name=='posix':
    # Ubuntu
    font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    font_name = fm.FontProperties(fname=font_path, size=10).get_name()
    print(font_name)
    plt.rc('font', family=font_name)
    mpl.font_manager._rebuild()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.callbacks import PerplexityMetric
from gensim.models.wrappers import DtmModel
from gensim.matutils import corpus2csc
from scipy.spatial.distance import cosine
from scipy.stats import linregress
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline
 
# Enable logging for gensim - optional
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
# # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.root.level = logging.INFO

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


### Dtm 클래스 ###
# class Dtm(DtmModel):
#     @classmethod

def term_variance(model, topic):
    """Finds variance of probability over time for terms for a given topic.
    High variance terms are more likely to be interesting than low variance
    terms."""
    p = np.exp(model.lambda_[topic]) /\
        np.exp(model.lambda_[topic]).sum(axis=0)
    variances = np.var(p, axis=1)
    order = np.argsort(variances)[::-1]
    terms = np.array([term for term, _
                    in sorted(model.id2word.token2id.items(),
                            key=lambda x: x[1])])[order]
    variances = variances[order]
    return list(zip(terms, variances))

def term_slope(model, topic):
    """Finds slope of probability over time for terms for a given topic.
    This is useful for roughly identifying terms that are rising or
    declining in popularity over time."""
    p = np.exp(model.lambda_[topic]) /\
        np.exp(model.lambda_[topic]).sum(axis=0)
    slopes = np.apply_along_axis(
        lambda y: linregress(x=range(len(y)), y=y).slope, axis=1, arr=p)
    order = np.argsort(slopes)
    terms = np.array([term for term, _
                    in sorted(model.id2word.token2id.items(),
                                key=lambda x: x[1])])[order]
    slopes = slopes[order]
    return list(zip(terms, slopes))

def topic_summary(model,topic, n=20):
    """Prints the top N terms by variance, increasing slope, and decreasing
    slope."""
    print('Variance\n---------')
    for row in term_variance(model,topic)[:n]:
        print(row)
    slopes = term_slope(model,topic)
    print('\nSlope (positive)\n----------')
    for row in slopes[-n:][::-1]:
        print(row)
    print('\nSlope (negative)\n----------')
    for row in slopes[:n]:
        print(row)

def plot_terms(model, topic, terms,times, title=None, name=None, hide_y=False):
    """Creates a plot of term probabilities over time in a given topic."""
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    colors = cm.jet(np.linspace(0,1,len(terms)))
    # for term in terms:
    #     ax.plot(
    #     # list(range(2000,2020)), # 연도 범위 2000~2019
    #     ['Jan','Feb','Mar'],
    #     term_distribution(model,term, topic),
    #     label=term)
    for idx, term in enumerate(terms):
        ax.plot(
        list(range(len(model.time_slices))), # t 범위
        term_distribution(model,term, topic),
        label=term,
        color=colors[idx]) # 단어 갯수 만큼 서로 다른 색상 부여

    # plt.xticks(np.array(['Jan','Feb','Mar'])) #2019년 x축에 추가
    plt.xticks(np.arange(len(model.time_slices)),np.array(times),rotation=45)
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if hide_y:
        ax.set_yticklabels([])
    ax.set_ylabel('Probability')
    if title:
        ax.set_title(title)
    if name:
        fig.savefig(
            name+'.png', dpi=500, bbox_extra_artists=(leg,), bbox_inches='tight')
        # pd.concat([pd.Series([model.show_topic(topic,0,20)[x][1] for x in range(20)],name='Jan'),
        #     pd.Series([model.show_topic(topic,1,20)[x][1] for x in range(20)],name='Feb'),
        # pd.Series([model.show_topic(topic,2,20)[x][1] for x in range(20)],name='Mar')],axis=1).to_excel(
        #     name+'.xlsx',index=None)
        topn_terms = pd.DataFrame()
        for idx, time in enumerate(times):
            res_topn_terms = pd.Series([model.show_topic(topic,idx,20)[x][1] for x in range(20)],name=time)
            topn_terms = pd.concat([topn_terms,res_topn_terms],axis=1)
        topn_terms.to_excel(name+'.xlsx',index=None,sheet_name='topic '+str(topic))
    #ax.set_axis_bgcolor('white')
    return fig, ax

def term_distribution(model, term, topic):
    """Extracts the probability over each time slice of a term/topic pair."""
    word_index = model.id2word.token2id[term]
    topic_slice = np.exp(model.lambda_[topic]) #model.lambda_는 [topic,words,time]에 따른 값.
    topic_slice = topic_slice / topic_slice.sum(axis=0)
    return topic_slice[word_index]

def summary(model, slices, topn=10):
    """Prints a summary of all the topics"""
    for topic in range(model.num_topics):
        print('Topic %d' % topic)
        print(model.top_term_table(topic, slices, topn))
        print()

def top_term_table(model, topic, slices, topn=10):
    """Returns a dataframe with the top n terms in the topic for each of
    the given time slices."""
    data = {}
    for time_slice in slices:
        time = np.where(model.time_slice_labels == time_slice)[0][0]
        data[time_slice] = [
            term for p, term
            in model.show_topic(topic, time=time, topn=topn)
        ]
    return pd.DataFrame(data)

def topic_distribution(model, time):
    """Extracts the probability over each time slice of a topic. """
    topic_slice = np.exp(model.lambda_[:,:,time]) #model.lambda_는 [topic,words,time]에 따른 값.
    topic_slice = topic_slice / topic_slice.sum(axis=0) #topic_slice.sum(axis=0) -> 시기별 토픽 內 전체 단어들의 확률 합
    topic_slice = topic_slice.sum(axis=1) / topic_slice.sum(axis=1).sum() 
    return topic_slice

# 시기별 토픽 비중변화 추출
def topic_proportion(model,year,time_slice):
    """ model = DTM 모델명, year = 시기(20년치라면 20), time_slice = DTM time_slice"""
    time_slice = np.insert(time_slice,0,0) # 첫번째 칸에 0 삽입
    topic_proportion = pd.DataFrame()
    for i in range(year):
        res = pd.DataFrame(model.gamma_[time_slice[i]:time_slice[i]+time_slice[i+1],].mean(axis=0)).T
        topic_proportion = topic_proportion.append(res,ignore_index=True)
    return topic_proportion