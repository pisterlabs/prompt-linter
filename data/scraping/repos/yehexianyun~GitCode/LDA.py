
import os
import pandas as pd

folder_path2 = "G:\\12_Database\\新闻数据"

# 获取文件夹中的所有excel文件
file_list = [f for f in os.listdir(folder_path2) if f.endswith('.xlsx')]

# 创建一个空的dataframe
df2 = pd.DataFrame()

# 循环读取每个excel文件并将其添加到dataframe中
for file in file_list:
    file_path = os.path.join(folder_path2, file)
    temp_df2 = pd.read_excel(file_path)
    temp_df2.columns = ['date', 'title', 'source']
    df2= pd.concat([df2, temp_df2], ignore_index=True)


# 设置文件夹路径
folder_path = "G:\\12_Database\\股吧文本数据"

# 获取文件夹中的所有pkl文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 创建一个空的dataframe
df = pd.DataFrame()

# 循环读取每个pkl文件并将其添加到dataframe中
for file in file_list:
    file_path = os.path.join(folder_path, file)
    temp_df = pd.read_excel(file_path)
    df = pd.concat([df, temp_df], ignore_index=True)

df = df.iloc[:, [4, 6]]
df.columns = ['date', 'title']
data = pd.concat([df, df2], ignore_index=True)
df3 = pd.DataFrame(data['title'].astype(str))
data.to_csv("G:\\12_Database\\LDA_data.csv")
df3 = pd.read_csv("G:\\12_Database\\LDA_data.csv")
#将df3中的title列转换为str类型
df3['title'] = df3['title'].astype(str)
#LDA model
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
 
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

import jieba
def chinese_word_cut(mytext):
    #将返回结果全部转换为str类型
    return " ".join(jieba.cut(mytext))

df3["content_cutted"] = df3.title.apply(chinese_word_cut) #分词,并且去掉
df3.content_cutted.head()
df3.to_csv("G:\\12_Database\\LDA_data.csv")

# 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(df3.content_cutted)

from sklearn.decomposition import LatentDirichletAllocation
n_topics = 8 
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50,
                                random_state=0)


lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

n_top_words = 20

tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, n_top_words)
'''
import pickle
pickle.dump(lda, open("lda_model.pkl", "wb"))
pickle.dump(tf_vectorizer, open("tf_vectorizer.pkl", "wb"))
pickle.dump(tf, open("tf.pkl", "wb"))
pickle.dump(df3, open("df3.pkl", "wb"))
pickle.dump(tf_feature_names, open("tf_feature_names.pkl", "wb"))
pickle.dump(n_features, open("n_features.pkl", "wb"))
pickle.dump(n_topics, open("n_topics.pkl", "wb"))
pickle.dump(n_top_words, open("n_top_words.pkl", "wb"))

import pickle
lda = pickle.load(open("lda_model.pkl", "rb"))
tf_vectorizer = pickle.load(open("tf_vectorizer.pkl", "rb"))
df3 = pickle.load(open("df3.pkl", "rb"))
tf_feature_names = pickle.load(open("tf_feature_names.pkl", "rb"))
n_features = pickle.load(open("n_features.pkl", "rb"))
n_topics = pickle.load(open("n_topics.pkl", "rb"))
n_top_words = pickle.load(open("n_top_words.pkl", "rb"))
tf = pickle.load(open("tf.pkl", "rb"))
'''
import pyLDAvis
import pyLDAvis.sklearn
import gensim
pyLDAvis.enable_notebook()
pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.display(pic)
pyLDAvis.save_html(pic, 'lda_pass'+str(n_topics)+'.html')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pyLDAvis.enable_notebook()
pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.display(pic)
pyLDAvis.save_html(pic, 'lda_pass'+str(n_topics)+'.html')

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)


import matplotlib.pyplot as plt

plexs = []
n_max_topics = 9
for i in range(1,n_max_topics):
    print(i)
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50,random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))


n_t=8#区间最右侧的值。注意：不能大于n_max_topics
x=list(range(1,n_t))
plt.plot(x,plexs[1:n_t])
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.show()
#保存图片
plt.savefig('perplexity.png',dpi=300,bbox_inches='tight')
