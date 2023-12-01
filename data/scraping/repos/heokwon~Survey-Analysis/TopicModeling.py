import nltk
from nltk.util import ngrams
from nltk import word_tokenize
from nltk import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, MLEProbDist
import numpy as np
import codecs
from tqdm import tqdm
import random
import numpy as np
import re
import pandas as pd
from konlpy.tag import Okt
from future.utils import iteritems
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import matplotlib as mpl
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import pyLDAvis.gensim_models
import gensim
import numpy as np
import spacy
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
# from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
# 워닝 메세지 삭제
import warnings
from datetime import datetime
from time import time
warnings.filterwarnings(action='ignore')

# # Colab 의 한글 폰트 설정
# plt.rc('font', family='NanumSquareRound') 

# 유니코드에서  음수 부호설정
mpl.rc('axes', unicode_minus=False)

# Seaborn 테마 설정
sns.set_theme(style="ticks", color_codes=True, font='NanumSquare', font_scale=2)

## font 경로설정
font_list = [font.name for font in fm.fontManager.ttflist]
font_list
# font_path = r'C:\Users\heokwon\anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\NanumGothic.ttf'

okt = Okt()

## 불용어 사전
stop_words = '수, nan, 없다, P, 없음, 등, 않은, 이, 의, 있, 하, 것, 들, 그, 되, 보, 않, 없, 나, 주, 같, 때, 도, 을, 는, 가, 에, 햄버거'
stop_words =  stop_words.split(', ')

## 전처리
def clean_content(content):
  cleaned = re.sub('[^가-힣ㄱ-ㅣa-zA-Z|.%]', ' ', string=str(content))
  cleaned = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', cleaned)
  return cleaned

## 한단어 제거
def oneWordRemoval(lst):
  for el in lst:
    if len(el) <= 1:
      lst.remove(el)

## 불용어 제거
def del_stop(data):
  words = []
  for w in data:
    if w not in stop_words:
      words.append(w)
  return words

def topic_modeling(excel_path, column_name, topic_nums, epochs, font_path):
    df=pd.read_excel(excel_path)
    df[column_name] = df[column_name].apply(clean_content)
    df = df[column_name].apply(okt.morphs, stem=True)
    df.apply(oneWordRemoval)
    df = df.apply(del_stop)
    
    ## TF-IDF
    dictionary = gensim.corpora.Dictionary(df)
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in df]
    tfidf = gensim.models.TfidfModel(BoW_corpus, smartirs='ntc')

    num = 0
    for doc in tfidf[BoW_corpus]:
      for id, freq in doc:
        num += 1
    
    tfidf_dic = {dictionary.get(id): freq for doc in tfidf[BoW_corpus] for id, freq in doc}
    od = {v: k for k, v in tfidf_dic.items()}
    ordered_keys = sorted(od.keys())
    tfidf_od = {od[k]: k for k in ordered_keys}

    keys = list(tfidf_od.keys())
    keys.reverse()
    for i in range(50):
      key = keys[i]

    ## LDA 학습
    chunksize = 2000 # 한번에 처리할 row 수 설정
    iterations = 400 # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
    eval_every = None # log 복잡도 계산 수행 유무 설정 1로 설정하면 학습이 2배로 느려짐
    random_state = 100 # 재실행 시 같은 결과를 같기 위해 난수 설정

    lda_model = gensim.models.ldamodel.LdaModel(corpus=tfidf[BoW_corpus],
                                                id2word=dictionary,
                                                num_topics=topic_nums,
                                                chunksize=chunksize,
                                                passes=epochs,
                                                iterations=iterations,
                                                random_state=random_state,
                                                eval_every=eval_every,)
    
    return lda_model, tfidf[BoW_corpus]
    
    ## topic modeling
def topic_wordcloud(lda_model,font_path, topic_nums):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0,
                      font_path=font_path)

    topics = lda_model.show_topics(formatted=False)

    if topic_nums == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)
    elif topic_nums == 3:
        fig, axes = plt.subplots(1, 3, figsize=(20,6), sharex=True, sharey=True)
    elif topic_nums == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15,7), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()