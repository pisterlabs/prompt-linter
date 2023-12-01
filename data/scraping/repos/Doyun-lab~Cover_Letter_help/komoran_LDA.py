import numpy as np
import pandas as pd
from konlpy.tag import Komoran  
from gensim import corpora, models
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel

komoran = Komoran()

# 데이터 불러오기
data = pd.read_csv("jobkorea_data.csv")

data.shape
data.columns.tolist()

x_list = data['답변']
x_list

# 명사 추출
data_word = []
for i in range(len(x_list)):
    try:
        data_word.append(komoran.nouns(x_list[i]))
    except Exception as e:
        continue

Data_list = x_list.values.tolist()

id2word = corpora.Dictionary(data_word)

# 20회 이하로 등장한 단어는 삭제
id2word.filter_extremes(no_below=0)

texts = data_word
corpus = [id2word.doc2bow(text) for text in texts]

mallet_path = './Downloads/mallet-2.0.8/bin/mallet' 
ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()

# coherence 계산
def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=data_word, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=21, step=2)

# 파라미터 설정
limit = 21
start = 4
step = 2
x = range(start, limit, step)

topic_num = 0
count = 0
max_coherence = 0

# 토픽 수 및 Coherence Value 출력
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", cv)
    coherence = cv
    if coherence >= max_coherence:
        max_coherence = coherence
        topic_num = m
        model_list_num = count   
    count = count+1

# 모델 선택과 토픽 출력
optimal_model = model_list[model_list_num]
model_topics = optimal_model.show_topics(formatted=False)
model_topics