import numpy as np
import logging
import json
import pandas as pd
import warnings
import jieba
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
# Import dataset
if __name__ == '__main__':
    df = pd.read_csv('../data/buwenminglvke.csv',header=None,sep=',',encoding='GBK').astype(str)
    #从文件导入停用词表
    stpwrdpath = "stop_words.txt"
    stpwrd_dic = open(stpwrdpath,encoding='GBK')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()


    #处理输入数据
    segment =[]
    for index,row in df.iterrows():
        content = row[7]
        if content != 'nan':
            words = jieba.cut(content)
            splitedStr=''
            rowcut=[]
            for word in words:
                if word not in stpwrdlst:
                    splitedStr += word + ' '
                    rowcut.append(word)
            segment.append(rowcut)
    docs=segment    #赋值给docs
    print(docs)
    dictionary = Dictionary(docs) #生成字典
    #dictionary.filter_extremes(no_below=10, no_above=0.2) #字典筛选
    print(dictionary)
    corpus = [dictionary.doc2bow(doc) for doc in docs] #生成语料库
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print(corpus[:5])

    #开始做LDA训练
    num_topics = 3
    chunksize = 500
    passes = 20
    iterations = 400
    eval_every = 1
    # Make a index to word dictionary.
    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token
    lda_model = LsiModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                            num_topics=num_topics)
    # Print the Keyword in the 5 topics
    print(lda_model.print_topics())
    print(lda_model.get_topics())
    #计算主题一致性指标
    coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_npmi')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    #绘制图像，K与一致性得分
    def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model= LsiModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                            num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, \
                                            texts=texts, \
                                            dictionary=dictionary, \
                                            coherence='c_uci')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values
    limit=8; start=2; step=1;#K的最大值，起始值，步长
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=start, limit=limit, step=step)
    # Show graph
    import matplotlib.pyplot as plt
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()