#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from konlpy.tag import Okt #형태소 분석
from nltk import bigrams
from nltk.util import ngrams
import pandas as pd

from gensim.models.ldamodel import LdaModel # LDA 분석
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel # 토픽갯수를 검증하기 위함
from tqdm.notebook import tqdm
from gensim import corpora
from gensim.models import TfidfModel

import warnings
warnings.filterwarnings(action='ignore')

okt=Okt()

#원하는 태그 추출 & 불용어 제거
def okt_pos_tagging(string):
    
    #불용어
    stopwords = pd.read_csv('ko-stopwords.csv')
    stopwords=list(stopwords['stopwords'])
    stopwords.extend(['에서','고','이다','는','한','씨', "것","거","게","데","이다","건","고","되다","되어다","걸","기",
                      "시","네","듯","랍니","중이","얘","스","도도", "나","수","개","내","기","제","저","인","있다","이렇다",
                      "그렇다","번","위","팅","분","인","링","란","포","두", "진짜", "하다" ,"이다" ,"가다", "이제" ,"들다",
                     "너무", "먹다"])
    stopwords=set(stopwords)    
    pos_words = okt.pos(string, stem=True, norm=True)
    words = [word for word, tag in pos_words if tag in ['Noun', 'Adjective', 'Verb'] if word not in stopwords ]
    return words


#bigram 만드는 함수
def bigram_(tokens):
    
    #document 별 형태소분석기로 분리된 토큰들을 bigram으로 변환(토큰은 string 상태)
    bigram_stack = [] #변환된 bigram
    for token in tokens:
        bigram = bigrams(token)
        bigram_token = [' '.join(grams) for grams in bigram]
        bigram_stack.append(bigram_token)
        
        
    #bigram  딕셔너리 생성(아이디 매칭)
    id2word = corpora.Dictionary(bigram_stack) #unigram
#     id2word.token2id  #아이디가 매칭된 딕셔너리 확인용
        
    
    
    #딕셔너리에 매칭한 corpus 생성
    #생성한 bigram 딕셔너리에서 bigram_stack의 토큰을 아이디로 매칭변환
    texts = bigram_stack 
    corpus = [id2word.doc2bow(text) for text in texts]
    
    
    #gensim으로 tf-idf 처리 #바이그램 요소 하나를 하나의 워드로 처리해서 tf_idf 점수
    tfidf = TfidfModel(corpus) 
    corpus_tfidf = tfidf[corpus]
    tfidf_corpus = [x for x in corpus_tfidf]
    return tfidf_corpus, corpus ,id2word




#perplexcity 계산(그래프 생성, 테이블 생성)
def perplexcity_graph(topic_n, bigram_result):
    #topic_n _ 몇개 토픽을 테스트 해볼것인지 지정    
    #perplexcity
    perplexity_value = []
    num_t = []
    for i in topic_n:
        model = LdaModel(bigram_result[0], num_topics = i, id2word=bigram_result[2])
        num_t.append(i)
        perplexity_value.append(model.log_perplexity(bigram_result[1]))

    #그래프
    plt.title('Perpelxity score')
    plt.plot(topic_n, perplexity_value)
    plt.xlabel('number of topics')
    plt.ylabel('perplexity_scores')
    plt.show
    plt.savefig('Perplexity_Graph.png')  #그래프저장
    
    
    #테이블 저장
    perplexity_df = pd.DataFrame(list(zip(num_t,perplexity_value)), columns = ['cluster','perplexity_score'])
    perplexity_df.to_csv('Perplexity_score_table.csv', encoding='utf-8-sig')
    perplexity_df
    
    
    
    
    
#coherence 계산(그래프 생성, 테이블 생성)    
def coherence_graph(topic_n, bigram_result):
    coherence_value = []
    num_t = []

    for i in topic_n:
        model = LdaModel(bigram_result[0], num_topics = i, id2word=bigram_result[2])
        cm = CoherenceModel(model=model, corpus=bigram_result[1], coherence='u_mass')
        coherence = cm.get_coherence()
        coherence_value.append(coherence)
        num_t.append(i)

    #그래프
    plt.title('Coherence score')
    plt.plot(topic_n, coherence_value)
    plt.xlabel('number of topics')
    plt.ylabel('coherence_scores')
    plt.show
    plt.savefig('Coherence_Graph.png')  #그래프저장

    
    #테이블 저장
    coherence_df = pd.DataFrame(list(zip(num_t,coherence_value)), columns = ['cluster','coherence_score'])
    coherence_df.to_csv('Coherence_score_table.csv', encoding='utf-8-sig')
    coherence_df
    
    
    
    
    
    
    
    
#doc 문서별 토픽 추출하기
def exctract_topic(ldamodel, corpus):
    topic_n = []
    topic_prop = []
    all_topic_prop = []
    for topic_doc in ldamodel[corpus]:
        topic_doc = sorted(topic_doc, key=lambda x: (x[1]), reverse=True) #여러 토픽이 할당되어 있는경우 높은 확률의 토픽순으로 정렬
        
        #확률 기준대로 정렬 했으므로 topic_doc[0] 제일 높은 확률의 토픽
        topic_n.append(int(topic_doc[0][0]))
        topic_prop.append(round(topic_doc[0][1],4))
        all_topic_prop.append(topic_doc)
        
    #데이터프레임화    
    data = {'topic':topic_n, 'topic_weight' : topic_prop ,'topic_all_weight': all_topic_prop}
    df = pd.DataFrame(data)
    return df