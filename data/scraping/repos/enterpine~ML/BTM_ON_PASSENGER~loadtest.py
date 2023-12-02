import pandas as pd
from gensim.corpora.dictionary import Dictionary
from BTM_ON_PASSENGER.BTMModel import BtmModel
from gensim.models.coherencemodel import CoherenceModel
def main():
    df=pd.read_table("../data/Laptops_Train.xml.seg")

    stpwrdpath = "stopwords_en.txt"
    stpwrd_dic = open(stpwrdpath, encoding='GBK')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    segment = []

    for i in range(len(df)):
        if i%3==2:
            if (i-3>=0 and df.loc[i].values[0].replace("$T$",df.loc[i+1].values[0])!=df.loc[i-3].values[0].replace("$T$",df.loc[i-2].values[0])) or i<3:

                sentence = df.loc[i].values[0].replace("$T$",df.loc[i+1].values[0])
                print(sentence)
                words=sentence.split(" ")

                rewords = []
                for word in words:
                    if word.lower() not in stpwrdlst:
                        rewords.append(word.lower())

                segment.append(rewords)


    docs = segment  # 赋值给docs ，每行数据分词

    print(docs)

    dictionary = Dictionary(docs)  # 生成字典 无序号字典['IPAD', '使用', '劝阻', '听', '机组人员']..

    BTMdic = {}  # 有序号字典 {'IPAD': 1, '使用': 2, '劝阻': 3, '听': 4, '机组人员': 5,....}
    for i in dictionary:
        BTMdic[dictionary[i]] = i + 1

    # 训练模型
    BitM = BtmModel(docs=docs, dictionary=BTMdic, topic_num=6, iter_times=10, alpha=0.1, beta=0.01, has_background=False)
    BitM.runModel()  # save(BitM)#BitM = load()
    BitM.show()  # 每个主题下，某单词出现的此数
    print(BitM.get_topics())

    # 计算一致性得分
    coherence_model_lda = CoherenceModel(model=BitM, texts=docs, dictionary=dictionary, coherence='c_npmi')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # # 不同主题数下的一致性得分变化数据
    # def compute_coherence_values(dictionary, texts, start, limit, step):
    #     coherence_values = []
    #     model_list = []
    #     for num_topics in range(start, limit, step):
    #         model = BtmModel(docs=docs, dictionary=BTMdic, topic_num=num_topics, iter_times=4, alpha=0.1, beta=0.01,
    #                          has_background=False)
    #         model.runModel()
    #         model_list.append(model)
    #         coherencemodel = CoherenceModel(model=model, \
    #                                         texts=texts, \
    #                                         dictionary=dictionary, \
    #                                         coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())
    #
    #
    #     return model_list, coherence_values
    #
    # limit = 100;
    # start = 2;
    # step = 10;  # K的最大值，起始值，步长
    # model_list, coherence_values = compute_coherence_values(dictionary=dictionary, texts=docs,
    #                                                         start=start, limit=limit, step=step)
    # # 绘制上图
    # import matplotlib.pyplot as plt
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values,label="c_v")
    #
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    # print("####")
    # input()

if __name__ == "__main__":
	main()