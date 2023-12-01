import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("cleaned_tweet_gen_remove_emoji_v4.csv")
#data = pd.read_csv("SaudiIrony.csv")
data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values


model = BERTopic.load("model_twitter_all_v3")
#loaded_model = BERTopic.load("model_twitter_all_v4")
output = model.get_topic(49)
print(output)

freq = model.get_topic_info()
freq.head(10)

freq = model.get_topic_info()
print("Number of topics: {}".format(len(freq)))
freq.head()


a_topic = freq.iloc[1]["Topic"] # Select the 1st topic
model.get_topic(a_topic) # Show the words and their c-TF-IDF scores

similar_topics, similarity = model.find_topics("politics", top_n = 3)

most_similar = similar_topics[0]
print("Most Similar Topic Info: \n{}".format(model.get_topic(most_similar)))
print("Similarity Score: {}".format(similarity[0]))

#fig = loaded_.visualize_topics()
#fig.write_html("path/to/file.html")

#loaded_model.visualize_topics()
#loaded_model.visualize_hierarchy()

# -1 is the outlayer is not clustered
model.get_topic_info().head(10)

# remove the 3-fix topics
topic_lables = model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=15, separator=" - ")
model.get_topic_info(topic_lables)


# create you own label
#model.set_topic_labels(0: "reinformecemnt leatning", 8: Transformer models"})
#model.get_topic_info().head(10)

# need data (document)
# Change n-gram without re-fitting our entire model
#model.update_topics(document, n_gram_range=(1, 3))

# Merge sepcific topics 
#model.merge_topics(documents, topics_to_mergs=[1,8,12])

# Reduce the number of topics by iteratively merging them 
#model.reduce_topics(document, nr_topics=100)

# Find specific topics 
#model.find_topics("politics", top_n=1)

# Topic Word Scores 
fig_chart = model.visualize_barchart(width=280, height=330, top_n_topics=8, n_words=10)
fig_chart.write_html("visual_chart.html")    

# heatmap plot
fig_heatmap = model.visualize_heatmap(n_clusters=20)
fig_heatmap.write_html("visual_heatmap.html") 

# hierarchy plot 
fig_hier = model.visualize_hierarchy()
fig_hier.write_html("visual_hierarchy.html") 

#model.visualize_documents(
#    documents,
#    reduced_embeddings=reduce_embeddings,
#    topics=list(range(30)),
#    custom_labels= True,
#    height=600
#)

#text = ["كورونا"]

#topics, probabilites = model.transform(text)

#print(topics)
