## 01_데이터 셋 입력
import pandas as pd

rawdata = pd.read_csv('rawdata_abstract.csv', encoding='cp949')
documents = pd.DataFrame(rawdata)

len(documents)

#%%
## 02_데이터 전처리 A
# 데이터 전처리 함수 import
import re # 문자열 정규 표현식 패키지
from nltk.corpus import stopwords
import nltk
from gensim.parsing.preprocessing import preprocess_string

nltk.download('stopwords')

def clean_text(d):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', d)
    return text

def clean_stopword(d): # 
    stop_words = stopwords.words('english')
    stop_words.extend(['disaster'])
    return ' '.join([w.lower() for w in d.split() if w.lower() not in stop_words and len(w) > 3])

def preprocessing(d):
    return preprocess_string(d)

#%%
## 02_데이터 전처리 B
# 영어 외 특수문자, 숫자 등 의미 추출 어려운 내용 제거
documents.replace("", float("NaN"), inplace=True)
# documents.isnull().values.any()
documents.dropna(inplace=True)
len(documents)

#%%
## 02_데이터 전처리 C
# 특수문자, 숫자 등 불용어 제거
# 토큰화 및 리스트로 변경
documents['keword'] = documents['keword'].apply(clean_stopword)
tokenized_docs = documents['keword'].apply(preprocessing)
tokenized_docs = tokenized_docs.tolist()

#%%
## 02_데이터 전처리 D
# 토큰의 개수가 5보다 작은 것을 삭제
import numpy as np
drop_docs = [index for index, sentence in enumerate(tokenized_docs) if len(sentence) <=5]
docs_texts = np.delete(tokenized_docs, drop_docs, axis = 0)
len(docs_texts)

#%%
## 03_LDA 토픽 모델링 A
from gensim import corpora
from gensim.models import LdaModel

dictionary = corpora.Dictionary(docs_texts)
corpus = [dictionary.doc2bow(text) for text in docs_texts]

lda_model = LdaModel(corpus, num_topics = 10, id2word=dictionary)
topics = lda_model.print_topics()
topics

#%%
## 03_LDA 토픽 모델링 B
# 최적 주제 선정을 위한 코드
from gensim.models.coherencemodel import CoherenceModel

min_topics, max_topics = 2, 10 ## (1) 3, 15 -> 12 /(2) 2, 10 -> 7
coherence_scores = []

for num_topics in range(min_topics, max_topics):
    model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence = CoherenceModel(model=model, texts=docs_texts, dictionary=dictionary)
    coherence_scores.append(coherence.get_coherence())
    
coherence_scores

#%%
## 03_LDA 토픽 모델링 C
# 코오런스 시각화
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10, 6))
plt.plot(x, coherence_scores)
plt.xlabel('number of topics')
plt.ylabel('coherence_scores')
plt.show()

#%%
## 03_LDA 토픽 모델링 D
from gensim.models import LdaModel

lda_model = LdaModel(corpus, num_topics = 12, id2word=dictionary)
topics = lda_model.print_topics()
topics

#%%
## 04_LDA 토픽 모델링 시각화
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)

pyLDAvis.save_html(vis, 'test.html')