from flask import Flask, Blueprint, render_template
from flask import request


import numpy as np
import pandas as pd
import warnings # 경고 메시지 무시
warnings.filterwarnings(action='ignore')
# 한국어 형태소 분석기 중 성능이 가장 우수한 Mecab 사용
# from konlpy.tag import Mecab
# mecab = Mecab()
from tqdm import tqdm # 작업 프로세스 시각화
import re # 문자열 처리를 위한 정규표현식 패키지
from gensim import corpora # 단어 빈도수 계산 패키지
import gensim # LDA 모델 활용 목적
# import pyLDAvis # LDA 시각화용 패키지
from collections import Counter # 단어 등장 횟수 카운트
from gensim.models.coherencemodel import CoherenceModel
import pickle
from gensim import corpora, models, similarities
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--title', type=str)
parser.add_argument('--book', type=bool, default=True)
parser.add_argument('--article', type=bool, default=True)
args = parser.parse_args()

def result():
    name = args.title
    book = args.book
    article = args.article
    # load data
    vectors = pd.read_json("../../data/CBF/final_tokens_2.json")
    df1 = pd.read_json('../../data/CBF/df_book_clean.json')
    df2 = pd.read_json('../../data/CBF/article_sum.json')

    df1 = df1.rename(columns={'_id':'id'})
    df_des = pd.concat([df1[['id','name_x','description']].rename(columns={'name_x':'title'}),df2[['id','title','content_tag_removed']].rename(columns={'content_tag_removed':'description'})], axis=0)
    df_des = df_des.reset_index().reset_index().rename(columns={'level_0':'문서 번호'})

    # Tokenize
    des_tokenized = []
    with open('../../data/CBF/des_tokenized.pkl','rb') as f:
        des_tokenized = pickle.load(f)

    entri_token = []
    for doc in vectors['tokens']:
        entri_token.append(doc)

    from gensim import corpora
    dictionary = corpora.Dictionary(entri_token) # 명사 집합들 사전화
    corpus = [dictionary.doc2bow(text) for text in des_tokenized] # 각 문서마다 각 명사의 갯수 분석
    
    num_topics = 35
    lda_model_final = models.LdaModel.load('../../data/CBF'+'/models6/ldamodels_bow_'+str(num_topics)+'.lda')
    corpus_lda_model = lda_model_final[corpus]
    index = similarities.MatrixSimilarity(lda_model_final[corpus])

    def book_recommender_book(title, book=True, article=True):
        books_checked = 0
        for i in range(len(df_des)):
            recommendation_scores = []
            # 넣은 타이틀이 동일할 경우
            if df_des.loc[i,'title'] == title:
                # i 번째 topic들 불러오기
                lda_vectors = corpus_lda_model[i]
                # 해당 토픽들 모임에 해당하는 similar matrix 값
                sims = index[lda_vectors]
                sims = list(enumerate(sims))
                for sim in sims:
                    book_num = sim[0]# enumerate index 값
                    recommendation_score = [df_des.iloc[book_num,2],df_des.iloc[book_num,3], sim[1]]
                    recommendation_scores.append(recommendation_score)
                
                if (book == True) and (article==True) :
                    recommendation = sorted(recommendation_scores, key=lambda x: x[2], reverse=True) # sim score 값에 따라 정렬
                    print("Your book's most prominent tokens are:")
                    article_tokens = corpus[i] # 해당 문서의 단어 토큰들
                    sorted_tokens = sorted(article_tokens, key=lambda x: x[1], reverse=True) # 단어 토큰의 빈도로 정렬
                    sorted_tokens_10 = sorted_tokens[:10]
                    for i in range(len(sorted_tokens_10)):
                        print("Word {} (\"{}\") appears {} time(s).".format(sorted_tokens_10[i][0], 
                                                                    dictionary[sorted_tokens_10[i][0]], 
                                                                    sorted_tokens_10[i][1]))
                    print('-----')
                    print("Your book's most prominant topic is:")
                    print(lda_model_final.print_topic(max(lda_vectors, key=lambda item: item[1])[0]))
                    print('-----')
                    print('Here are your recommendations for "{}":'.format(title))
                    for i in recommendation[1:15]:
                        print(i[1],i[2])  
                    return recommendation
            
                elif (book == True) and (article == False) :
                    recommendation = sorted(recommendation_scores[:373], key=lambda x: x[1], reverse=True) # sim score 값에 따라 정렬
                    print("Your book's most prominent tokens are:")
                    article_tokens = corpus[i] # 해당 문서의 단어 토큰들
                    sorted_tokens = sorted(article_tokens, key=lambda x: x[2], reverse=True) # 단어 토큰의 빈도로 정렬
                    sorted_tokens_10 = sorted_tokens[:10]
                    for i in range(len(sorted_tokens_10)):
                        print("Word {} (\"{}\") appears {} time(s).".format(sorted_tokens_10[i][0], 
                                                                    dictionary[sorted_tokens_10[i][0]], 
                                                                    sorted_tokens_10[i][1]))
                    print('-----')
                    print("Your book's most prominant topic is:")
                    print(lda_model_final.print_topic(max(lda_vectors, key=lambda item: item[1])[0]))
                    print('-----')
                    print('Here are your recommendations for "{}":'.format(title))
                    for i in recommendation[1:15]:
                        print(i[1],i[2])   
                    return recommendation       

                elif (book == False) and (article == True) :
                    recommendation = sorted(recommendation_scores[373:], key=lambda x: x[2], reverse=True) # sim score 값에 따라 정렬
                    print("Your book's most prominent tokens are:")
                    article_tokens = corpus[i] # 해당 문서의 단어 토큰들
                    sorted_tokens = sorted(article_tokens, key=lambda x: x[1], reverse=True) # 단어 토큰의 빈도로 정렬
                    sorted_tokens_10 = sorted_tokens[:10]
                    for i in range(len(sorted_tokens_10)):
                        print("Word {} (\"{}\") appears {} time(s).".format(sorted_tokens_10[i][0], 
                                                                    dictionary[sorted_tokens_10[i][0]], 
                                                                    sorted_tokens_10[i][1]))
                    print('-----')
                    print("Your book's most prominant topic is:")
                    print(lda_model_final.print_topic(max(lda_vectors, key=lambda item: item[1])[0]))
                    print('-----')
                    print('Here are your recommendations for "{}":'.format(title))
                    for i in recommendation[1:15]:
                        print(i[1],i[2])
                    return recommendation
             
                else :
                    print("sorry, you should select either book or article or both")
            else:
                books_checked +=1
        
        # 만약 for문을 다돌았는데 못찾았을 경우
        if books_checked == len(df_des): 
            book_suggestions = []
            print('Sorry, but it looks like "{}" is not available.'.format(title))

    recommendation_book = book_recommender_book(name,book,article)

    return recommendation_book


if __name__ == "__main__":
    result()
