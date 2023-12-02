import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sklearn

#1、预处理数据#
df = pd.read_csv('../data/buwenminglvke.csv',header=None,sep=',',encoding='GBK').astype(str)
# 2、分词

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
segment =[]  #存储分词结果
for index,row in df.iterrows():
    content = row[7]
    if content != 'nan':
        #print(content)
        words = jieba.cut(content)
        splitedStr=''
        #print(words)
        for word in words:
            if word not in stpwrdlst:
                splitedStr += word + ' '
        segment.append(splitedStr)

#print(segment)

cntVector = CountVectorizer()
cntTf = cntVector.fit_transform(segment)

#print(cntTf)

lda = LatentDirichletAllocation(n_topics=3,learning_offset=50,random_state=1)
docres = lda.fit_transform(cntTf)
print (docres)

def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print
    #打印主题-词语分布矩阵
    print (model.components_)

n_top_words = 7
tf_feature_names = cntVector.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

###############################################################################################
###############################################################################################
###############################################################################################
from gensim.models.coherencemodel import CoherenceModel
lda_model = lda
docs=segment
dictionary=cntVector.get_feature_names()

print(dictionary)

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)