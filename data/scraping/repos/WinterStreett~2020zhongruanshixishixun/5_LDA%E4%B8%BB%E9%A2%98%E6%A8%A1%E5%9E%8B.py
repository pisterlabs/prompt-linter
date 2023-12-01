# -*- coding = 'utf-8' -*-
__author__ = '七月小组'
'''使用LDA主题模型对评论数据进行主题分析，并可视化（云图）
   本程序用到的两个数据：正面评论和负面评论，使用ROST软件的情感分析功能得到
'''
#导入库
import pandas as pd
from gensim import corpora, models
import re
import numpy as np
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.misc import imread


def plt_wc(dict, impath, fontpath="./data/SIMYOU.TTF", max_font_size=40, random_state=30, max_words=500, min_font_size=2):
    '''根据输入的词典，制作词云图'''
    pic = imread(impath)
    wc = WordCloud(background_color='white',
                   mask =pic,
                   max_font_size=max_font_size,
                   random_state=random_state,
                   font_path=fontpath,
                   max_words=max_words,
                   min_font_size=min_font_size)
    wc.fit_words(dict)
    plt.figure(figsize=(60,50))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


#步骤1
#读取正面和负面评论数据
positive_comment_path = u'./data/comment_data_正面情感结果.txt'
negative_comment_path = u'./data/comment_data_负面情感结果.txt'
#先用open方法打开文件，直接使用pd.read_csv读取名字有中文名的文件会报错
fp = open(positive_comment_path, 'r', encoding='utf-16le')
fn = open(negative_comment_path, 'r', encoding='utf-16le')
positive_comment = pd.read_csv(fp, header=None)
negative_comment = pd.read_csv(fn, header=None)
#步骤2
#去除前缀,并按空格将每一行字符串分成列表
positive_comment = pd.DataFrame(positive_comment[0].apply(lambda s:re.sub(r'^.*\t', '', s)))#去除数字和制表符的前缀
negative_comment = pd.DataFrame(negative_comment[0].apply(lambda s:re.sub(r'^.*\t', '', s)))
positive_comment = pd.DataFrame(positive_comment[0].apply(lambda s:s.strip()))#去除两端的空字符
negative_comment = pd.DataFrame(negative_comment[0].apply(lambda s:s.strip()))
positive_comment = pd.DataFrame(positive_comment[0].apply(lambda s:s.split(' ')))#按空格分割
negative_comment = pd.DataFrame(negative_comment[0].apply(lambda s:s.split(' ')))
#步骤3
#制作词典，词语向量化
positive_dict = corpora.Dictionary(positive_comment[0])
negative_dict = corpora.Dictionary(negative_comment[0])
#步骤4
#制作语料：将给给定语料（词典）转换为词袋模型
positive_corpus = [positive_dict.doc2bow(i) for i in positive_comment[0]]#建立语料库
negative_corpus = [negative_dict.doc2bow(i) for i in negative_comment[0]]
#步骤5
#LDA模型训练,并对参数进行优化&选择
np.random.seed(1)
num_topic_list = [1,2,3,4,5,6,7,8,9,10]
print('正面模型num_topics参数比较...')
positive_cm_list = []
for topic_num in num_topic_list:
    positive_LDA = models.LdaModel(positive_corpus, num_topics=topic_num, id2word=positive_dict, iterations=1000, alpha='auto')
    #利用CoherenceModel方法对模型进行评估
    cm = CoherenceModel(model=positive_LDA, corpus=positive_corpus, dictionary=positive_dict, coherence='u_mass')
    positive_cm_list.append(cm.get_coherence())
print('负面模型num_topics参数比较...')
negative_cm_list = []
for topic_num in num_topic_list:
    negative_LDA = models.LdaModel(negative_corpus, num_topics=topic_num, id2word=negative_dict, iterations=1000, alpha='auto')
    cm = CoherenceModel(model=negative_LDA, corpus=negative_corpus, dictionary=negative_dict, coherence='u_mass')
    negative_cm_list.append(cm.get_coherence())
plt.figure()
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.plot(num_topic_list, positive_cm_list,label='正面模型')
plt.plot(num_topic_list, negative_cm_list,label='负面模型')
# plt.plot(num_topic_list, positive_cm_list,num_topic_list, negative_cm_list)
plt.xlabel('num_topics',fontsize=15)
plt.ylabel('cm',fontsize=15)
plt.title('不同主题数模型的cm值连线图',fontsize=20)
plt.legend()
plt.show()
#步骤6
#由步骤5确定证明和负面LDA模型的主题数，开始训练模型
positive_num_topics = 3
negative_num_topics = 1
positive_LDA = models.LdaModel(positive_corpus, num_topics=positive_num_topics, id2word=positive_dict, iterations=1000, alpha='auto')
negative_LDA = models.LdaModel(negative_corpus, num_topics=negative_num_topics, id2word=negative_dict, iterations=1000, alpha='auto')
#步骤7
#将得到的LDA模型分析结果可视化
top_num = 80#取前80个主题词
for i in range(positive_num_topics):
    '''对于每一个正面主题，提取主题词和对应的权值，形成键值对，以此画出云图'''
    result = positive_LDA.print_topic(i, topn=top_num)#result是一个字符串，包含了主题词和权值
    positive_topic_words = re.findall(r'"(.*?)"', result)#匹配主题词
    positive_topic_score = []
    positive_topic_score.append(result[:5])# 先通过切片拿到第一个主题词的权重，这样可以使下面的正则表达式简单一些
    positive_topic_score += re.findall(r'\+(.*?)\*', result)  # 匹配主题词权重
    positive_topic_score = [float(i) for i in positive_topic_score]  # 将权重从字符串转换为浮点数
    dict_word_score = dict(zip(positive_topic_words, positive_topic_score))
    plt_wc(dict_word_score, './data/timg.gif')
for i in range(negative_num_topics):
    '''对于每一个负面主题，提取主题词和对应的权值，形成键值对，以此画出云图'''
    result = negative_LDA.print_topic(i, topn=top_num)#result是一个字符串，包含了主题词和权值
    negative_topic_words = re.findall(r'"(.*?)"', result)#匹配主题词
    negative_topic_score = []
    negative_topic_score.append(result[:5])# 先通过切片拿到第一个主题词的权重，这样可以使下面的正则表达式简单一些
    negative_topic_score += re.findall(r'\+(.*?)\*', result)  # 匹配主题词权重
    negative_topic_score = [float(i) for i in negative_topic_score]  # 将权重从字符串转换为浮点数
    dict_word_score = dict(zip(negative_topic_words, negative_topic_score))
    plt_wc(dict_word_score, './data/timg.gif')