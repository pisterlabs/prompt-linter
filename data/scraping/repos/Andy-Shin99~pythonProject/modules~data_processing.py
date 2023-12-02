#https://velog.io/@mare-solis/LDA-%ED%86%A0%ED%94%BD-%EB%AA%A8%EB%8D%B8%EB%A7%81%EC%9C%BC%EB%A1%9C-%EC%BD%98%ED%85%90%EC%B8%A0-%EB%A6%AC%EB%B7%B0%EB%A5%BC-%EB%B6%84%EC%84%9D%ED%95%98%EC%9E%90
import re
from hanspell import spell_checker
from konlpy.tag import Komoran
import collections
from collections import Counter
from itertools import combinations
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric

import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def text_clean(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern, '', text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거    
    text = re.sub(pattern, '', text)
    pattern = '([a-zA-Z0-9]+)'   # 알파벳, 숫자 제거  
    text = re.sub(pattern, '', text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern, '', text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern, '', text)
    return text  

def pairwise(arr):
    toks = list(dict.fromkeys(arr))
    pair = list(combinations(sorted(toks), 2))
    return pair

def preprocessing_data(duration) :
    title_data = []
    des_data = []
    data = [title_data, des_data]

    filename = './/results//corpus_data//' + duration + '.txt'
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        title_data.append(line.strip())
        if len(title_data) >= 1000 :
            break
    for line in f :
        des_data.append(line.strip())
    f.close()

    title_corpus = []
    des_corpus = []
               
    for title in title_data :
        corpus = text_clean(title)
        title_corpus.append(corpus)

    for des in des_data :
        corpus = text_clean(des)
        des_corpus.append(corpus)


    title_corpus = list(map(lambda x : spell_checker.check(x).checked, title_corpus)) #['제목', '제목' ...]
    des_corpus = list(map(lambda x : spell_checker.check(x).checked, des_corpus)) #['내용', '내용' ...]
  
    return title_corpus, des_corpus

def get_title_result(duration, corpus):
    title_tokens = []

    for sentence in corpus:
        words = komoran.nouns(sentence)
        tokens = []
        for word in words :
            if word not in stopwords :
                tokens.append(word)
        title_tokens.append(tokens)

    title_pairs = []

    for token in title_tokens :
        pair = pairwise(token)
        title_pairs += pair

    count = collections.Counter(list(title_pairs))
    count = {k: v for k, v in count.items() if v >= 20} #threshold

    scount = sorted(count.items(), key=lambda x:x[1], reverse=True)
    
    filename = './/results//title_results//' + str(duration)[:4] + '.txt'
    with open(filename, 'w', encoding='utf-8') as f:    
        for k, v in scount:
            f.write(f'{k} : {v}\n')
    f.close()

    return

def get_des_result(duration, corpus) :
        tokens = []
        for sentence in corpus:
            sentence_word = []
            words = komoran.nouns(sentence)
            for word in words :
                if word not in stopwords :
                    sentence_word.append(word)
            tokens.append(sentence_word)
        '''
        count = Counter(tokens)

        tag_count = []
        tags = []

        for n, c in count.most_common(50) :
            dics = {'tag': n, 'count': c}
            tag_count.append(dics)
            tags.append(dics['tag'])
        
        h = open(duration[:4] | '.txt', 'w')
        for tag in tag_count :
            h.write(" {:<14}".format(tag['tag']), end = '\t')
            h.write("{}".format(tag['count']))
        h.close()
        '''

        #토픽모델링
        dictionary = corpora.Dictionary(tokens)
        dictionary.filter_extremes(no_below=20, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        temp = dictionary[0]
        id2word = dictionary.id2token
        if duration[2:4] == '20' | duration[2:4] == '22' :
            num_topics=3

        #2020, 2022: num_topics=4

        model = LdaModel(
         corpus=corpus,
        id2word=id2word,
        num_topics=num_topics, #생성될 토픽의 개수
        chunksize=1000, #한번의 트레이닝에 처리될 문서의 개수
        passes=50, #전체 코퍼스 트레이닝 횟수
        iterations=400, #문서 당 반복 횟수
        eval_every=None
)
        top_topics = model.top_topics(corpus)
        #lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
        #filename = './/results//topic_modeling_results//' + duration[:4] + '.html'
        #pyLDAvis.save_html(lda_visualization, filename)

        


duration_list = ['20160101to20161231', '20170101to20171231', '20180101to20181231', '20190101to20191231', '20200101to20201231', '20210101to20211231', '20220101to20221231', '20230101to20231110']

stopwords = []
stopwords = stopwords + [line.strip() for line in open('./dics/stopwordsKor.txt', encoding='utf-8')]

komoran = Komoran(userdic='.//dics//user_dic.txt')

for duration in duration_list :
    title_corpus, des_corpus = preprocessing_data(duration)
    get_title_result(duration, title_corpus)
    #get_des_result(duration, des_corpus)


