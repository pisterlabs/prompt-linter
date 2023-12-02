import jieba
import pandas as pd
import jieba.posseg as posseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import pyLDAvis
import matplotlib
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud
from googletrans import Translator
import time
from tqdm import tqdm
import pyecharts.options as opts
from pyecharts.charts import WordCloud


def translate_chinese_to_english(text):
    translator = Translator(service_urls=['translate.google.com.hk'])
    translation = translator.translate(text, src='zh-cn', dest='en')
    return translation.text


def tf_idf(df):
    corpus = []
    for i in df['new_content']:
        corpus.append(i.strip())

        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}
    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)

    x_data = list(df2['word'])[:30]
    y_data = list(df2['tfidf'])[:30]
    x_data1 = [translate_chinese_to_english(x) for x in x_data]
    x_data1.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9), dpi=500)
    plt.barh(x_data1, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("TF-IDF TOP30")
    plt.xlabel("value")
    plt.savefig('TFIDF.png')
    plt.show()
    df2.to_csv('tfidf.csv', encoding='utf-8-sig', index=False)


def wordclound(df):
    str1 = ''
    counts = {}
    for i in df['new_content']:
        res = posseg.cut(i)
        for word, flag in res:
            if len(word) >= 2:
                counts[word] = counts.get(word, 0) + 1
                str1 += word + ' '

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data[1:201]
    x_data1 =[]
    for i in tqdm(df1['word']):
        i = translate_chinese_to_english(i)
        x_data1.append(i)
        time.sleep(0.1)
    df1['translate'] = x_data1
    df1['counts'] = y_data[1:201]
    df1.to_csv('TOP200热词.csv', encoding="utf-8-sig")

    data = []
    for x,y in zip(x_data1,y_data[1:201]):
        data.append((x,y))

    (
        WordCloud()
            .add(series_name="Hot spot analysis", data_pair=data, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Hot spot analysis", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("wordcloud.html")
    )
    # str1 = str1.strip(' ')
    # # str1 = translate_chinese_to_english(str1)
    # stylecloud.gen_stylecloud(text=str1, max_words=200,
    #                           collocations=False,
    #                           background_color="#B3B6B7",
    #                           font_path='simhei.ttf',
    #                           icon_name='fas fa-tree',
    #                           size=500,
    #                           palette='matplotlib.Inferno_9',
    #                           output_name='词云图.png')
    # Image(filename='词云图.png')


def lda(df):
    train = []
    for line in df['new_content']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    # 构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    #
    # # 困惑度模块
    # x_data = []
    # y_data = []
    # z_data = []
    # for i in tqdm(range(2, 15)):
    #     x_data.append(i)
    #     lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
    #     # 困惑度计算
    #     perplexity = lda_model.log_perplexity(corpus)
    #     y_data.append(perplexity)
    #     # 一致性计算
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
    #     coherence = coherence_model_lda.get_coherence()
    #     z_data.append(coherence)
    #
    # # 绘制困惑度和一致性折线图
    # fig = plt.figure(figsize=(15, 5))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    #
    # # 绘制困惑度折线图
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.plot(x_data, y_data, marker="o")
    # plt.title("perplexity_values")
    # plt.xlabel('num topics')
    # plt.ylabel('perplexity score')
    # #绘制一致性的折线图
    # ax2 = fig.add_subplot(1, 2, 2)
    # plt.plot(x_data, z_data, marker="o")
    # plt.title("coherence_values")
    # plt.xlabel("num topics")
    # plt.ylabel("coherence score")
    #
    # plt.savefig('困惑度和一致性.png')
    # plt.show()
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['主题数'] = x_data
    # df5['困惑度'] = y_data
    # df5['一致性'] = z_data
    # df5.to_csv('困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    # LDA可视化模块
    # 构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111,
                                          iterations=400)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, 'lda.html')


if __name__ == '__main__':
    df = pd.read_csv('new_data.csv')
    df.dropna(subset=['new_content'], axis=0,inplace=True)
    # tf_idf(df)
    wordclound(df)
    # lda(df)