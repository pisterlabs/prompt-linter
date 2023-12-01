import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import ldamodel
from multiprocessing import freeze_support
import pyLDAvis.gensim
import pyLDAvis
from tqdm import tqdm
import warnings
import os
import re
# 忽略所有警告
warnings.filterwarnings("ignore")


def demo(df1,time_name):
    data = df1
    # 构建词典
    corpus = []
    # 读取预料 一行预料为一个文档
    for d in data['clearn_comment']:
        d = str(d).split(" ")
        corpus.append(d)

    dictionary = Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]

    # # 定义评估指标函数
    # def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    #     coherence_values = []
    #     model_list = []
    #     # 使用tqdm显示进度条
    #     for num_topics in tqdm(range(start, limit, step)):
    #         model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    #         model_list.append(model)
    #         #Coherence（一致性）是用来评估主题模型的一种指标，c_v方法使用了类似点互信息的计算，可以有效地衡量主题的连贯性
    #         coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherence_model.get_coherence())
    #
    #     return model_list, coherence_values
    #
    # # 调用评估函数来计算不同主题数下的模型评估指标
    # start = 2
    # limit = 16
    # step = 1
    # model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_bow, texts=corpus,start=start, limit=limit, step=step)
    #
    # # 绘制评估指标随主题数变化的曲线
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.title('{}_coherence_values'.format(time_name))
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence Score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.savefig('./{}/coherence_values.png'.format(time_name))
    # plt.show()
    # df = pd.DataFrame()
    # df['Topic number'] = x
    # df['coherence_values'] = coherence_values
    # df.to_csv('./{}/coherence_values.csv'.format(time_name),encoding='utf-8-sig')
    # 根据coherence值选择最优主题数
    # optimal_index = np.argmax(coherence_values)
    # optimal_model = model_list[optimal_index]
    # optimal_num_topics = start + optimal_index * step
    # print("Optimal number of topics:", optimal_num_topics)

    optimal_num_topics = 15

    # LDA可视化模块
    # 构建lda主题参数
    lda = ldamodel.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=optimal_num_topics)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus_bow, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, './{}/lda.html'.format(time_name))

    # 主题判断模块
    list3 = []
    list2 = []
    # 这里进行lda主题判断
    for i in lda.get_document_topics(corpus_bow)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    data = pd.DataFrame()
    data['主题概率'] = list3
    data['主题类型'] = list2

    #获取对应主题出现的频次
    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []
    list_word = []
    #根据其对应的词，来获取其相应的权重
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    #把上面权重的词计算好之后，进行保存为csv文件
    df2 = pd.DataFrame()
    for j,k,l in zip(topic,list_gailv1,list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('./{}/主题词分布表.csv'.format(time_name), encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./{}/特征词.csv'.format(time_name),encoding='utf-8-sig',index=False)

    return optimal_num_topics


if __name__ == '__main__':
    df1 = pd.read_csv('new_data.csv')
    df2 = pd.DataFrame()
    df2['time'] = df1['createdAt']
    df2['content'] = df1['message']
    df2['clearn_comment'] = df1['clearn_comment']
    df2['情感类型'] = df1['情感类型']
    df2['情感得分'] = df1['情感得分']

    df3 = pd.read_csv('new_data1.csv')
    df4 = pd.DataFrame()
    df4['time'] = df3['created_at']
    df4['content'] = df3['text']
    df4['clearn_comment'] = df3['clearn_comment']
    df4['情感类型'] = df3['情感类型']
    df4['情感得分'] = df3['情感得分']

    df5 = pd.read_csv('new_data3.csv')
    df6 = pd.DataFrame()
    df6['time'] = df5['createdAt']
    df6['content'] = df5['message']
    df6['clearn_comment'] = df5['clearn_comment']
    df6['情感类型'] = df5['情感类型']
    df6['情感得分'] = df5['情感得分']

    df7 = pd.read_csv('new_data4.csv')
    df8 = pd.DataFrame()
    df8['time'] = df7['created_at']
    df8['content'] = df7['text']
    df8['clearn_comment'] = df7['clearn_comment']
    df8['情感类型'] = df7['情感类型']
    df8['情感得分'] = df7['情感得分']

    data = pd.concat([df2, df4,df6,df8], axis=0)
    data = data.drop_duplicates(subset=['content'])

    def time_process(x):
        x1 = str(x).split(" ")
        x2 = x1[0] + " " + x1[1] + " " + x1[-1]
        return x2

    data['time'] = data['time'].apply(time_process)
    data['time'] = pd.to_datetime(data['time'])
    data.index = data['time']
    new_df = data['time'].value_counts()
    new_df = new_df.sort_index()
    x_data = [str(x) for x in new_df.index]

    x_data1 = []
    y_data1 = []
    for x in x_data:
        x = str(x).split(" ")
        x = x[0]
        x_data1.append(x)
        if not os.path.exists("./{}".format(x)):
            os.mkdir("./{}".format(x))
        data1 = data[x:x]
        optimal_num_topics = demo(data1,x)
        y_data1.append(optimal_num_topics)

    df = pd.DataFrame()
    df['日期'] = x_data1
    df['最优主题数'] = y_data1
    df.to_csv('time_lda.csv',encoding='utf-8-sig')