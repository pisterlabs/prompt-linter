# Utilize topic models to analyze text across various time periods.
import jieba
import numpy as np
import pandas as pd
import glob, os
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import data
path = 'C:\学习\python\Thesis\Scraping'
#path = 'C:\学习\python\Thesis\Scraping'
os.chdir(path) # directory of text files
filelist = glob.glob("*.txt")
text=[]
doc_len=[]
for file in filelist:
    with open(file,'r',encoding='utf-8') as f:
        text.append(list(set(list(i.replace('\n', '').replace(' ', '').replace('\t', '') for i in list(f.readlines())))))
#with open(path+'all.txt','r',encoding='utf-8') as f:
    #text=list(set(list(i.replace('\n', '').replace(' ', '').replace('\t', '') for i in list(f.readlines()))))
        #doc_len.append(len(open(file, 'r',encoding='utf-8').read()))
#print(text)
#print(doc_len)
#plt.hist(doc_len, bins=1019, color='navy')
#plt.xlabel('Length')
#plt.ylabel('Frequency')
#plt.title('Lengths of documents')
#plt.xlim([0, 8000])
#plt.savefig('length')
#plt.show()

# Cut word
# Obtain stopwords
def stopwordslist():
    stopwords = [line.strip() for line in open('C:/Users/过青灯客/others/stopwords.txt', 'r', encoding='UTF-8').readlines()]
    return stopwords
# Cut each line
def word_cut(line):
    wcut = jieba.cut(line.strip())
    stopwords = stopwordslist()
# Exclude stopwords and other confusing components
    remain = ''
    for word in wcut:
        if word not in stopwords:
            if len(word) > 1:
                if word != '\t':
                    if word!= '\u3000':
                        #remain.append(word)
                        remain += word
                        remain += " "
    return remain
# Result of cutting
result_cut = []
for sublist in text:
    for i in sublist:
        if word_cut(i) != '':
            #print(line_cut(i))
            result_cut.append([i, word_cut(i)])
result_cut = [i[1].split(' ')[:-1] for i in result_cut]
print(result_cut)
# Create dictionary: create ID for each word in the list of word-cutting
id2word = corpora.Dictionary(result_cut)
# Turn dictionary into corpus, which count words using words' ids
corpus = [id2word.doc2bow(word) for word in result_cut]
#print(corpus[:1])
# [[(1,3),(2,1)],[]] What is returned is a list consisting of sublists. Sublists contain tuples. (1,3) represents word 1 appears 3 times in the first document.
#print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])# Use this to look up words' ids in the dictionary

# Create LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                id2word=id2word,
                num_topics=input("Number of topics:"),
                # random_state=100,
                # update_every=1,
                # chunksize=100,
                # passes=10,
                # alpha='auto',
                # per_word_topics=True
                )
pprint(lda_model.print_topics(100, num_words=20))

# Find the dominant topic in each document
def dominant_topic_of_doc(ldamodel=lda_model, corpus=corpus, text=text):
    dom_topic_of_doc_df = pd.DataFrame() # Create an empty dataframe
    for i, row in enumerate(ldamodel[corpus]): #ldamodel[corpus] is like [(1, 0.266), (2,0.933), (topic number, percentage of the topic)]
        row = sorted(row, key=lambda x: (x[1]), reverse=True) #The result is like [(2,0.933),(1, 0.266)], which puts the topic with the highest percentage first
#for i, row in enumerate(lda_model[corpus]):
    #row = sorted(row, key=lambda x: (x[1]), reverse=True)
#pprint(list(enumerate(row)))
        for j, (topic_num, percentage_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                dom_topic_word_percentage = ldamodel.show_topic(topic_num)
                dom_topic_keywords = ", ".join([word for word, percentage in dom_topic_word_percentage])
                dom_topic_of_doc_df = dom_topic_of_doc_df.append(pd.Series([int(topic_num), round(percentage_topic,4), dom_topic_keywords]), ignore_index=True)
            else:
                break
    dom_topic_of_doc_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    print(dom_topic_of_doc_df)
    # Add original text to the end of the output
    content = pd.Series(text)
    print(content)
    dom_topic_of_doc_df = pd.concat([dom_topic_of_doc_df, content], axis=1)
    return(dom_topic_of_doc_df)
dominant_topic_of_doc_df = dominant_topic_of_doc(ldamodel=lda_model, corpus=corpus, text=text)
# Format of dataframe
df_dominant_topic_of_doc = dominant_topic_of_doc_df.reset_index()
df_dominant_topic_of_doc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# Show
#print(df_dominant_topic_of_doc)
output='C:/Users/过青灯客/df_dominant_topic_of_doc.xlsx'
df_dominant_topic_of_doc.to_excel(output,index=True,header=True,encoding='utf_8_sig')

# Topic distribution across documents
# Number of Documents for Each Topic
doc_topic_counts = dominant_topic_of_doc_df ['Dominant_Topic'].value_counts()
# Percentage of Documents for Each Topic
topic_contribution = round(doc_topic_counts/doc_topic_counts.sum(), 4)
# Topic Number and Keywords
topic_num_keywords = dominant_topic_of_doc_df[['Dominant_Topic', 'Topic_Keywords']]
# Concatenate Column wise
df_topic_dist = pd.concat([topic_num_keywords, doc_topic_counts, topic_contribution], axis=1)
# Change Column names
df_topic_dist.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
# Show
print(df_topic_dist)
output1='C:/Users/过青灯客/df_topic_dist.xlsx'
df_topic_dist.to_excel(output1,index=True,header=True,encoding='utf_8_sig')
