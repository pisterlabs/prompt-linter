# util
import pandas as pd
import urllib.request
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# nltk 추가 다운
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# LDA
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models   # 시각화

# custom LDA 모델
class LDAmodel:
    def __init__(self, dataset, print=True, target_col='content'):
        self.dataset = pd.DataFrame({'Movie_name':dataset['Movie_name'],
                                     'content':dataset[target_col],
                                     'content_token':dataset[target_col]
                                    })
        self.model = None
        self.topic_table = None
        self.print = print  # flag to print

    def train(self, additional_stop_words=[], num_topics=70, passes=60, iterations=200, random_state=3569):
        self.additional_stop_words = additional_stop_words
        self.num_topics = num_topics    # n개의 토픽, k=n
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state

        # 이름(대문자) 추출 후, 불용어로 사용
        more_stop_words = []
        r = re.compile('[A-Z]{2,}')
        for idx in range(0, len(self.dataset)):
            more_stop_words = more_stop_words + r.findall(self.dataset.loc[idx]['content_token'])

        # 중복 제거
        newlist = []
        for x in more_stop_words:
            x = x.lower()
            if x not in newlist:
                newlist.append(x)
        more_stop_words = newlist

        # 전부 소문자로
        self.dataset['content_token'] = self.dataset['content_token'].str.lower()
        if self.print: print("[LDAmodel] 소문자 변경 완료")

        # 토큰화
        self.dataset['content_token'] = self.dataset.apply(lambda row: nltk.word_tokenize(row['content_token']), axis=1)
        if self.print: print("[LDAmodel] 토큰화 완료")

        # 불용어 제거
        stop_words = stopwords.words('english')
        stop_words = stop_words + more_stop_words + self.additional_stop_words

        self.dataset['content_token'] = self.dataset['content_token'].apply(lambda x: [word for word in x if word not in (stop_words)])
        if self.print: print("[LDAmodel] 불용어 제거 완료")

        # 표제어 추출
        self.dataset['content_token'] = self.dataset['content_token'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
        if self.print: print("[LDAmodel] 표제어 추출 완료")

        # 길이 3 이하 제거
        tokenized_doc = self.dataset['content_token'].apply(lambda x: [word for word in x if len(word) > 3])
        if self.print: print("[LDAmodel] 길이 3 이하 제거 완료")

        # TF-IDF 행렬 만들기
        # 역토큰화
        detokenized_doc = []
        for i in range(len(self.dataset)):
            t = ' '.join(tokenized_doc[i])
            detokenized_doc.append(t)

        # 다시 self.dataset['content_token']에 재저장
        self.dataset['content_token'] = detokenized_doc

        # 상위 1,000개의 단어를 보존
        vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000)
        X = vectorizer.fit_transform(self.dataset['content_token'])
        if self.print: print("[LDAmodel] TF-IDF 행렬 생성 완료")

        # TF-IDF 행렬의 크기 확인
        if self.print: print('[LDAmodel] TF-IDF 행렬의 크기 :', X.shape)

        # 정수 인코딩과 단어 집합 만들기
        self.dictionary = corpora.Dictionary(tokenized_doc)
        #self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_doc]
        if self.print: print("[LDAmodel] 정수 인코딩과 단어 집합 생성 완료")
        #print(len(corpus), corpus[1]) # 수행된 결과에서 두번째 출력. 첫번째 문서의 인덱스는 0

        # LDA 모델 훈련시키기
        self.model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes, iterations=self.iterations, random_state=self.random_state)
        if self.print: print('[LDAmodel] 학습 완료')

        #topics = self.model.print_topics(num_words=10)
        #for topic in topics:
        #   print(topic)

    # def visualization(self):
    #     pyLDAvis.enable_notebook()
    #     vis = pyLDAvis.gensim_models.prepare(self.model, self.corpus, self.dictionary)
    #     pyLDAvis.display(vis)
 
    def make_topic_table_per_doc(self):
        if self.model is None:
            print('[LDAmodel] plz train first')
            return

        self.topic_table = pd.DataFrame()
        min=999999
        max=-1
        # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
        for i, topic_list in enumerate(self.model[self.corpus]):
            doc = topic_list[0] if self.model.per_word_topics else topic_list            
            doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
            # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
            # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
            # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
            # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

            # 모든 문서에 대해서 각각 아래를 수행
            for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
                if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                    self.topic_table = self.topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                    if min > topic_num:
                        min = topic_num
                    if max < topic_num:
                        max = topic_num
                    # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
                else:
                    break
        if self.print: print('[LDAmodel] 생성 완료, 토픽 인덱스 범위 (' + str(min) + '-' + str(max) + ')')

        self.topic_table = self.topic_table.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
        self.topic_table.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
        return(self.topic_table)

    def n_doc_topic_list(self, n, num_words=10):
        if self.topic_table is None:
            print('[LDAmodel] plz make topic_table first')
            return

        topics = self.model.print_topics(num_topics=-1, num_words=num_words)
        topic_list = []
        for topic in topics:
            topic_list.append(topic[1].split("\"")[1::2])

        #print(len(topic_list))
        return(topic_list[int(self.topic_table.loc[n]['가장 비중이 높은 토픽'])])

    def evaluation(self):
        if self.model is None:
            print('[LDAmodel] plz train first')
            return

        cm = CoherenceModel(model=self.model, corpus=self.corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        return coherence, self.model.log_perplexity(self.corpus)    # coherence, perplexity
