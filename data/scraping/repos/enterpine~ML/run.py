import jieba
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from BTM_ORG_REMASTER.BTMModel import BtmModel
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel


def compute_coherence_values_lsa(dictionary, corpus, texts, start, limit, step,coherence='u_mass'):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LsiModel(corpus=corpus, id2word=dictionary.id2token, chunksize=chunksize, \
                         num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, \
                                        texts=texts, \
                                        dictionary=dictionary, \
                                        coherence=coherence)
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def compute_coherence_values_btm(dictionary, texts, start, limit, step,coherence='u_mass'):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = BtmModel(docs=docs, dictionary=BTMdic, topic_num=num_topics, iter_times=50, alpha=0.1, beta=0.01,
                         has_background=False)
        model.runModel()
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, \
                                        texts=texts, \
                                        dictionary=dictionary, \
                                        coherence=coherence)
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def compute_coherence_values_lda(dictionary, corpus, texts, start, limit, step,coherence='u_mass'):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word= dictionary.id2token, \
                       chunksize=chunksize,num_topics=num_topics, \
                       alpha=50/num_topics, eta=0.01,iterations=iterations, \
                       passes=passes, eval_every=eval_every)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, \
                                        texts=texts, \
                                        dictionary=dictionary, \
                                        coherence=coherence)
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

if __name__ == "__main__":
    df = pd.read_csv('../data/buwenminglvke.csv', header=None, sep=',', encoding='GBK').astype(str)
    stpwrdpath = "stop_words.txt"
    stpwrd_dic = open(stpwrdpath, encoding='GBK')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    # 处理输入数据
    segment = []
    for index, row in df.iterrows():
        content = row[7]
        if content != 'nan':
            words = jieba.cut(content)
            splitedStr = ''
            rowcut = []
            for word in words:
                if word not in stpwrdlst:
                    splitedStr += word + ' '
                    rowcut.append(word)
            segment.append(rowcut)
    docs = segment  # 赋值给docs


    dictionary = Dictionary(docs)  # 生成字典
    corpus = [dictionary.doc2bow(doc) for doc in docs]  # 生成语料库
    BTMdic = {}
    for i in dictionary:
        BTMdic[dictionary[i]] = i+1

    limit = 100;
    start = 2;
    step = 10;  # K的最大值，起始值，步长

    num_topics = 3
    chunksize = 500
    passes = 20
    iterations = 400
    eval_every = 1

    coherence_type='u_mass'#{'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
    model_list_lda, coherence_values_lda=compute_coherence_values_lda(dictionary=dictionary, corpus=corpus, texts=docs, start=start,limit=limit, step=step, coherence =coherence_type)
    model_list_btm, coherence_values_btm=compute_coherence_values_btm(dictionary=dictionary, texts=docs, start=start, limit=limit, step=step, coherence =coherence_type)
    model_list_lsa, coherence_values_lsa =compute_coherence_values_lsa(dictionary=dictionary, corpus=corpus,texts=docs, start=start, limit=limit, step=step,coherence =coherence_type)
    import matplotlib.pyplot as plt
    x = range(start, limit, step)
    plt.plot(x, coherence_values_lda,"x-",label="lda")
    plt.plot(x, coherence_values_btm,"+-",label="btm")
    plt.plot(x, coherence_values_lsa, "*-", label="lsa")
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score: "+coherence_type)
    plt.grid(True)
    plt.legend( loc='best')
    plt.show()