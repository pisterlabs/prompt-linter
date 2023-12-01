#!./.conda/envs/py310/bin/python3.10

# This is the tool file for topic model building
# author: Xinyi Li
# contact: xl4412@nyu.edu
# time of completion: 2023-07-20



import pandas as pd
import re
# import spacy  # 3.5.0
# from spacy import displacy
# from spacy.lang.zh.examples import sentences
import codecs


import pyLDAvis
import pyLDAvis.gensim as gensimvis
import gensim
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
#
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import jieba
# names
jieba.load_userdict("data/organization.txt")
jieba.load_userdict("data/organization2.txt")
jieba.load_userdict("data/person.txt")
# cities
jieba.load_userdict("data/glossary.txt")

class LDA():
    def __int__(self,data,loadPath,stopword_path):
        # self.stopwords = self.stopwordslist(stopPath)
        self.data = data
        self.loadPath = loadPath
        self.stopword_path = stopword_path

    @property
    def stopwordslist(self):
        '''
        load stopword list
        :return:
        '''
        stopwords = [line.strip() for line in open(self.stopword_path, 'r', encoding='utf-8').readlines()]
        # if you want to customize to add any stopword, uncomment the following lines:
        # stopwords.append('your target word')
        return stopwords


    def seg_sentence(self,sentence):
        '''
        # 对句子进行分词
        :return:cut word dict
        '''
        sentence = re.sub(r'[0-9\.]+', r'', sentence)
        sentence_seged = jieba.cut(sentence.strip())
        stopwords = self.stopwordslist  # 这里加载停用词的路径
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords and word.__len__() > 1:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr


    def save_word(self):
        '''
        save the word dict into file
        :param outPath:
        :return:
        '''
        outputs = open(self.outPath, 'w', encoding='utf-8')
        j=1
        for i in self.data['InfoTitle']:
            line_seg = self.seg_sentence(i)  # 这里的返回值是字符串
            outputs.write(line_seg + '\n')
            print('write in %d'%j)
            j = j+1

        outputs.close()

    @property
    def load_word(self):
        '''
        load the text file into train data set
        :param loadPath:
        :return: train data set, list consist of list
        '''
        train = []
        fp = codecs.open(self.loadPath,'r',encoding='utf8')
        for line in fp:
            if line != '':
                line = line.split()
                train.append([w for w in line])
        return train


    def LDA_model(self,num_topics,passes):
        '''
        train a LDA model
        :param loadPath:
        :return:
        '''

        dictionary = corpora.Dictionary(self.load_word)

        self.corpus = [dictionary.doc2bow(text) for text in self.load_word]

        # passes：训练伦次
        self.lda = LdaModel(corpus=self.corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
        print('LDA model trained.')
        return self.lda

    def word_count(self, is_ascend=False):
        all_words = pd.DataFrame({'all_words':eval(self.load_word)})
        words_count = all_words.groupby(by=['all_words'])['all_words'].agg([('count', 'count')])
        words_count = words_count.reset_index().sort_values(by=['count'], ascending=is_ascend).reset_index(drop=True)  # 降序
        return words_count

    def topic_visualization(self,model):
        '''
        3 ways to visualize the Topic words
        :return:
        '''

        # [Method_1]
        # 打印主题
        # for topic in model.print_topics(num_words = 20):
        #     termNumber = topic[0]
        #     print(topic[0], ':', sep='')
        #     listOfTerms = topic[1].split('+')
        #     for term in listOfTerms:
        #         listItems = term.split('*')
        #         print('  ', listItems[1], '(', listItems[0], ')', sep='')

        # [Method_2]
        for topic in model.print_topics(num_words=10):
            print(topic)

        # [Method_3]
        # Notice:This method requires specific version of pyLDAvis==2.1.2
        # use `pip install pyLDAvis=2.1.2 -i http://pypi.douban.com/simple --trusted-host` to download pyLDAvis

        # ldaCM = CoherenceModel(model=model, corpus=self.corpus, dictionary=dictionary, coherence='u_mass')
        # 用pyLDAvis将LDA模式可视化
        # pyLDAvis.enable_notebook()
        # plot = gensimvis.prepare(self.lda, self.corpus, corpora.Dictionary(self.train))
        # pyLDAvis.display(plot)
        # # 保存到本地html
        # pyLDAvis.save_html(plot, './pyLDAvis.html')

    def sentences_chart(self, start=0, end=10):
        corp = self.corpus[start:end]
        mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        fig, axes = plt.subplots(end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160)
        axes[0].axis('off')
        for i, ax in enumerate(axes):
            if i > 0:
                corp_cur = corp[i - 1]
                topic_percs, wordid_topics, wordid_phivalues = self.lda.get_document_topics(corp_cur,
                                                                                             per_word_topics=True)
                word_dominanttopic = [(self.lda.id2word[wd], topic[0]) for wd, topic in wordid_topics]
                ax.text(0.01, 0.5, "Doc " + str(i - 1) + ": ", verticalalignment='center',
                        fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

                # Draw Rectange
                topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
                ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                       color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

                word_pos = 0.06
                for j, (word, topics) in enumerate(word_dominanttopic):
                    if j < 14:
                        ax.text(word_pos, 0.5, word,
                                horizontalalignment='left',
                                verticalalignment='center',
                                fontsize=16, color=mycolors[topics],
                                transform=ax.transAxes, fontweight=700)
                        word_pos += .009 * len(word)  # to move the word for the next iter
                        ax.axis('off')
                ax.text(word_pos, 0.5, '. . .',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color='black',
                        transform=ax.transAxes)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2), fontsize=22,
                     y=0.95,
                     fontweight=700)
        plt.tight_layout()
        plt.show()

    def save_model(self,savePath,name):
        self.lda.save(savePath+name)