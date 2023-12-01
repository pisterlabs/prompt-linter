# -*- coding: utf-8 -*-

import pandas as pd
from konlpy.tag import Mecab
from tqdm import tqdm
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
import re
from LDA_score import get_score
import os
from api import *

mecab = Mecab("C:\\mecab\\mecab-ko-dic") # mecab dictionary 경로. colab에서 할 때는 안 넣어줘도 됐었음
mallet_path = "C:\\Mallet\\bin\\mallet"  # 이거 mallet2108어쩌구인가로도 바꿔보기


# 1. Preprocessing Data
def get_key_tokens(text):
    key_pos = ['SL', 'NNG', 'NNP', 'VV', 'VA', 'XR', 'SH'] # ['NNG', 'NNP', 'SL', 'SH']
    text = re.sub(r'\[[^)]*\]', '', text) # 한겨레 [포토], [인터뷰], [11회 비정규 노동 수기 공모전], [단독] 이런거 없애기
    tokens = mecab.pos(text)
    token_list = []
    for token, pos in filter(lambda x: (x[1] in key_pos), tokens):
        if pos == 'VV' or pos == 'VA' or pos == 'XR':
            if len(token) <= 1:
                continue
        token_list.append(token)
    #print(token_list)
    return token_list

  # return ','.join([token for token, pos in filter(lambda x: (x[1] in key_pos), tokens)])

def preprocess(file, nobelow):
    news_df = pd.read_csv(file)
    news_df_len = len(news_df.index)

    # get only the time, title data
    news_df = news_df[['_id', 'date', 'summary']]

    # chage data type
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['summary'] = news_df['summary'].astype(str)
    news_df['_id'] = news_df['_id'].astype(str)
    #2. Delete Stopwords
    # 2.1 tokenize & get the pos
    # 뉴스 데이터의 특징: 띄어쓰기, 오탈자 문제 적음.
    summary_list = []
    for i in tqdm(range(len(news_df['summary']))):
        summary_list.append(get_key_tokens(news_df.loc[i,'summary']))

    # 3. Make Corpus
    id2word = corpora.Dictionary(summary_list)
    id2word.filter_extremes(no_below=nobelow) 
    corpus = [id2word.doc2bow(content) for content in summary_list]

    return news_df, id2word, corpus, summary_list

# 2.2 delete stopwords => TODO

# 2.3 make user dictionary => TODO



# 4. Topic Modeling
def topic_modeling(id2word, corpus, summary_list, num_topics):

    # 얘 passes라는 인자로 epoch 조절 가능
    # ver 1.
    # lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=60, id2word=id2word) # 아래랑 성능 비교하기(너무 오래걸려;)
    # print(lda_model.print_topics(60, 3))
    # coherence_model = CoherenceModel(model=lda_model, texts=summary_list, dictionary=id2word, topn=10)
    # coherence = coherence_model.get_coherence() # 얘가 문제
    # print(coherence)

    # ver 2.
    # # 그 막 출력되는게 위에 코드 결과야 1000번 도는거 아마 1000번이 default인가봐
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, iterations=1000) # gensim 3.8 버전에만 존재
    pprint(ldamallet.show_topics(num_topics=num_topics, num_words=3))
    # # 4.1 get coherence
    #coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=summary_list, dictionary=id2word, coherence='c_v')
    #coherence_ldamallet = coherence_model_ldamallet.get_coherence()

    return ldamallet

# 4.3 formmating output with DF
def topics_to_csv(news_df, ldamodel, corpus, texts, num_keywords, num_topics):
    # Init output
    topics_info_df = pd.DataFrame()
    #print(1)
    # Get main topic in each document
    #ldamodel[corpus]: lda_model에 corpus를 넣어 각 토픽 당 확률을 알 수 있음
    for i, row in enumerate(ldamodel[corpus]):
        #print(2)

        # print(i, row) i는 전체 뉴스 기사 idx, row는 [(토픽 idx, 그 토픽일 확률), ... 토픽 개수만큰]
        row = sorted(row, key=lambda x: (x[1]), reverse=True) # 토픽일 확률 기준으로 sort
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                #print(3)
                wp = ldamodel.show_topic(topic_num, topn=num_keywords) #여기변경~~~~~~~~~~
                #print(4)
                topic_keywords = ", ".join([word for word, prop in wp])
                #print(5)
                #topics_info_df = pd.concat([topics_info_df, pd.Series([int(topic_num), round(prop_topic,4), topic_keywords])], axis=1)
                topics_info_df = topics_info_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    #print(6)
    topics_info_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    topics_info_df = pd.concat([topics_info_df, news_df['_id'], news_df['summary'],news_df['date']], axis=1)
    #print(7)
    topics_info_df = topics_info_df.reset_index()
    topics_info_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'ID', 'Summary', 'Date']

    topics_info_df['Dominant_Topic'] =topics_info_df['Dominant_Topic'] +1
    topics_info_df.Dominant_Topic = topics_info_df.Dominant_Topic.astype(str)
    #print(8)
    topics_info_df['Dominant_Topic'] =topics_info_df['Dominant_Topic'].str.split('.').str[0]
    #print(9)
    os.mkdir("\\result\\한겨레_summary_" + str(num_topics))
    #print(10)
    for i in range(1,num_topics+1):
        #print(11)
        globals()['df_{}'.format(i)]=topics_info_df.loc[topics_info_df.Dominant_Topic==str(i)]
        #print(12)
        globals()['df_{}'.format(i)].sort_values('Topic_Perc_Contrib',ascending=False,inplace = True)
        #print(13)
        globals()['df_{}'.format(i)].to_csv (".\\result\\한겨레_summary_" + str(num_topics) + "\\topic("+str(i)+")_news.csv", index = None)



if __name__ == '__main__':
    file = 'DM\\TopicModeling\\한겨레_코로나_요약.csv' # 경로 입력할 때 역슬래시 두개 넣기,,,
    
    news_df, id2word, corpus, summary_list = preprocess(file, 5) # 인자값 = no_below 값

    # find optimal topic nums
    #ldamallet, coherence_mallet = topic_modeling(id2word, corpus, summary_list)
    start = 40 # 이 범위는 뉴스 개수에 따라 다르게 하기
    limit = 101
    step = 5
    topic_priority = get_score(corpus, id2word, summary_list, start, limit, step)
    for tn in topic_priority:
        ldamallet = topic_modeling(id2word, corpus, summary_list, tn) #이렇게하려면 sort해서 주면 안 됨
        topics_to_csv(news_df, ldamallet, corpus, summary_list, 3, tn) # 마지막 인자 키워드 개수
    #model_list, coherence_values = compute_coherence_values(id2word, corpus, summary_list, start, limit, step)
    # find_optimal_model(model_list, coherence_values, start, limit, step)

    
    #print(topic_keywords_df)