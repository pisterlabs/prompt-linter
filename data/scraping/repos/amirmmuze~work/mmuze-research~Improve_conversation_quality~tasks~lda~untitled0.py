#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:38:51 2019

@author: amirdavidoff

eyal  + lda



todo:
    1. remove stop words
    2. stemming ? shorts > short
    2. search for a better way to cluster short text !

"""


''' eyal sdk additional info '''



sdk_path = '/Users/amirdavidoff/Desktop/data/yaron_sql_dumps/sdk_reports.csv'

spark_sdk  = sqlContext.read.options(header=True).csv(sdk_path) #,sep='\t'

additional_info = spark_sdk.filter(F.col("additional_info").contains(F.lit("/search"))).toPandas()

additional_info.to_csv('/Users/amirdavidoff/Desktop/data/additional_info.csv')

''' eyal text count '''
nlu = sqlContext.read.parquet('/Users/amirdavidoff/Desktop/data/enriched_data/nlu')


nlu.count()

nlu.columns


nlu_text = nlu.groupBy('text').count().toPandas()


nlu_text = nlu_text.sort_values('count',ascending=False).reset_index(drop=True)


nlu_text.to_csv('/Users/amirdavidoff/Desktop/data/nlu_raw_text_count.csv')




''' lda '''

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

nlu = sqlContext.read.parquet('/Users/amirdavidoff/Desktop/data/enriched_data/nlu')

docs = nlu.groupBy('sender_id').agg(F.collect_list('text').alias('texts_list')).toPandas()


rmv = ['yes','new search']



def clean_text(s):
    
    ls=[]
    for i in s:
        ls+=i.split()

    rm = ['yes','new','search']
    ls = [i for i in ls if i not in rm]
    
    ls = [i.lower() for i in ls if i.isalpha()]
    ls = [word for word in ls if word not in stopwords.words('english')]

        
    return ls


docs["splitted_text"] = docs["texts_list"].apply(clean_text)

docs["len"] = docs["splitted_text"].apply(len)


import gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel


not_empty = docs[docs["len"]!=0]



data_words = not_empty["splitted_text"].tolist()


# Build the bigram and trigram models
#bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
#trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
#bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)




id2word = corpora.Dictionary(data_words)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_words]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10000,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=10,
                                           per_word_topics=True)

import pickle 

pickle.dump(lda_model,open("/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/model.pickle","wb"))
pickle.dump(corpus,open("/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/corpus.pickle","wb"))

lda_model_loaded = pickle.load(open("/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/model.pickle","rb"))

print(lda_model_loaded.print_topics())



''' WORD CLOUD '''
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(#stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()




''' TSNE '''
# Get topic weights and dominant topics ------------s
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
#output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
plt.show()


''' viz '''

import pyLDAvis.gensim
#pyLDAvis.enable_notebook()

lda_model_loaded = pickle.load(open("/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/model.pickle","rb"))
corpus_loaded = pickle.load(open("/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/corpus.pickle","rb"))
                                 
                                

                        
vis = pyLDAvis.gensim.prepare(lda_model_loaded, corpus_loaded, dictionary=lda_model_loaded.id2word)
pyLDAvis.save_html(vis, '/Users/amirdavidoff/mmuze-research/Improve_conversation_quality/tasks/lda/vis.html')

