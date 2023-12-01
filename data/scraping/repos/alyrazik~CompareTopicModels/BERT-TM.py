import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from sklearn.datasets import fetch_20newsgroups

import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(name="BERT_logger")

# code from https://colab.research.google.com/drive/1OT_wcYKpKS73uR6y7IVYjJVxaP-C1H3k?usp=sharing#scrollTo=y_eHBI1jSb6i

data = pd.read_csv("C:\\Users\\Aly\\PycharmProjects\\TopicModel\\datasets\\arabic_dataset_classifiction.csv")
logger.info("Data read into memory ")
data = data.dropna()
documents = data['text'].values
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
logger.info("convert series to numpy array")
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-base-arabertv02')
logger.info("Define arabert")

model = BERTopic(language="english")
topics, probabilities = model.fit_transform(docs)

model.get_topic_freq().head()
model.get_topic(49)
model.visualize_topics()
#
# topic_model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False,
#                      embedding_model=arabert)
# topics, probs = topic_model.fit_transform(documents)
# topic_model.get_topic_freq().head(5)
# print(topic_model.get_topic(1)[:10])
# texts = [[word for word in str(document).split()] for document in documents]
# id2word = corpora.Dictionary(texts)
# corpus = [id2word.doc2bow(text) for text in texts]
# topics = []
# for i in topic_model.get_topics():
#   row = []
#   topic = topic_model.get_topic(i)
#   for word in topic:
#      row.append(word[0])
#   topics.append(row)
#   # compute Coherence Score
#
#   cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_npmi')
#   coherence = cm.get_coherence()
#   print('\nCoherence Score: ', coherence)
#   topic_model.visualize_topics()
#   # Save model
#   topic_model.save("my_model")
#   # Load model
#   my_model = BERTopic.load("my_model")
