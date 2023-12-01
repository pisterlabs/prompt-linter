import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from tqdm import tqdm

from collections import Counter
import itertools

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.models import LdaModel
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from nltk.stem import WordNetLemmatizer
import nltk

import pyLDAvis.gensim
import pyLDAvis

lemmatizer = WordNetLemmatizer()

stop_words = []
with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))


def processing_data(df):
    #去掉标点符号，以及机械压缩
    def preprocess_word(word):
        # Remove punctuation
        word = word.strip('\'"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word


    #再次去掉标点符号
    def gettext(x):
        import string
        punc = string.punctuation
        for ch in punc:
            txt = str(x).replace(ch,"")
        return txt


    #替换表情包
    def handle_emojis(tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
        # Angry -- >:(, >:-(, :'(
        tweet = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', tweet)
        # Surprised -- :O, :o, :-O, :-o, :0, 8-0
        tweet = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', tweet)
        # Confused -- :/, :\, :-/, :-\
        tweet = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', tweet)
        # Embarrassed -- :$, :-$
        tweet = re.sub(r'(:\\$|:-\\$)', ' ', tweet)
        # Other emoticons
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
        for emoticon in emoticons:
            tweet = tweet.replace(emoticon, " ")
        return tweet


    def clean_text(text):
        # Replaces URLs with the word URL
        text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
        # Replace @username with the word USER_MENTION
        text = re.sub(r'@[\S]+', ' ', text)
        # Replace #hashtag with the word HASHTAG
        text = re.sub(r'#(\S+)', ' ', text)
        text = re.sub(r'#\w+', ' ', text)
        # Remove RT (retweet)
        text = re.sub(r'\brt\b', ' ', text)
        # Replace 2+ dots with space
        text = re.sub(r'\.{2,}', ' ', text)
        # Strip space, " and ' from text
        text = text.strip(' "\'')
        # Handle emojis
        text = handle_emojis(text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
        # Lowercase and split into words
        words = text.lower().split()
        words = [w for w in words if w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        words = [preprocess_word(w) for w in words]
        if len(words) != 0:
            return ' '.join(words)
        else:
            return np.NAN

    df['text'] = df['text'].apply(gettext)
    df['text'] = df['text'].apply(preprocess_word)
    df['text'] = df['text'].apply(clean_text)
    df.dropna(subset=['text'],axis=0,inplace=True)
    return df


#LDA建模
def lda(df,name1):
    train = []
    for line in df['text']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]


    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
        # 困惑度计算
        perplexity = lda_model.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        z_data.append(coherence)

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    #绘制一致性的折线图
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")
    if not os.path.exists("./整体"):
        os.mkdir("./整体")
    plt.savefig('./整体/{}_困惑度和一致性.png'.format(name1))
    plt.show()
    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./整体/{}_困惑度和一致性.csv'.format(name1),encoding='utf-8-sig',index=False)

    optimal_z = max(z_data)
    optimal_z_index = z_data.index(optimal_z)
    best_topic_number = x_data[optimal_z_index]

    num_topics = best_topic_number

    #LDA可视化模块
    #构建整体参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './整体/{}_lda.html'.format(name1))

    # #主题判断模块
    # list3 = []
    # list2 = []
    # #这里进行整体判断
    # for i in lda.get_document_topics(corpus)[:]:
    #     listj = []
    #     list1 = []
    #     for j in i:
    #         list1.append(j)
    #         listj.append(j[1])
    #     list3.append(list1)
    #     bz = listj.index(max(listj))
    #     list2.append(i[bz][0])
    #
    # data = pd.DataFrame()
    # data['内容'] = df['分词']
    # data['主题概率'] = list3
    # data['主题类型'] = list2
    #
    # data.to_csv('./整体/lda_data.csv',encoding='utf-8-sig',index=False)

    # #获取对应主题出现的频次
    # new_data = data['主题类型'].value_counts()
    # new_data = new_data.sort_index(ascending=True)
    # y_data1 = [y for y in new_data.values]
    #
    # #主题词模块
    # word = lda.print_topics(num_words=20)
    # topic = []
    # quanzhong = []
    # list_gailv = []
    # list_gailv1 = []
    # list_word = []
    # #根据其对应的词，来获取其相应的权重
    # for w in word:
    #     ci = str(w[1])
    #     c1 = re.compile('\*"(.*?)"')
    #     c2 = c1.findall(ci)
    #     list_word.append(c2)
    #     c3 = '、'.join(c2)
    #
    #     c4 = re.compile(".*?(\d+).*?")
    #     c5 = c4.findall(ci)
    #     for c in c5[::1]:
    #         if c != "0":
    #             gailv = str(0) + '.' + str(c)
    #             list_gailv.append(gailv)
    #     list_gailv1.append(list_gailv)
    #     list_gailv = []
    #     zt = "Topic" + str(w[0])
    #     topic.append(zt)
    #     quanzhong.append(c3)
    #
    # #把上面权重的词计算好之后，进行保存为csv文件
    # df2 = pd.DataFrame()
    # for j,k,l in zip(topic,list_gailv1,list_word):
    #     df2['{}-主题词'.format(j)] = l
    #     df2['{}-权重'.format(j)] = k
    # df2.to_csv('./整体/主题词分布表.csv', encoding='utf-8-sig', index=False)
    #
    # y_data2 = []
    # for y in y_data1:
    #     number = float(y / sum(y_data1))
    #     y_data2.append(float('{:0.5}'.format(number)))
    #
    # df1 = pd.DataFrame()
    # df1['所属主题'] = topic
    # df1['文章数量'] = y_data1
    # df1['特征词'] = quanzhong
    # df1['主题强度'] = y_data2
    # df1.to_csv('./整体/特征词.csv',encoding='utf-8-sig',index=False)


# #绘制主题强度饼图
# def plt_pie():
#     plt.style.use('ggplot')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.figure(dpi=500)
#     if not os.path.exists("./整体"):
#         os.mkdir("./整体")
#     df = pd.read_csv('./整体/特征词.csv')
#     x_data = list(df['所属主题'])
#     y_data = list(df['文章数量'])
#     plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
#     plt.title('theme strength')
#     plt.tight_layout()
#     plt.savefig('./整体/theme strength.png')


if __name__ == '__main__':
    name = ['斯伦贝谢2018-2022','哈利伯顿2018-2022','贝克休斯2018-2022']
    for n in tqdm(name):
        df = pd.read_excel('三家公司2018-2022SCI论文数据汇总.xlsx',sheet_name=n)
        df['text'] = df['TI'] + df['SO'] + df['ID']
        data = processing_data(df)
        lda(data,n)

    name = ['斯伦贝谢2018-2022', '哈利伯顿2018-2022', '贝克休斯2018-2022']
    for n in tqdm(name):
        df = pd.read_excel('三家公司2018-2022SCI论文数据汇总.xlsx', sheet_name=n)
        df['text'] = df['TI'] + df['SO']
        data = processing_data(df)
        n = str(n) + '标题与关键词分析'
        lda(data, n)





