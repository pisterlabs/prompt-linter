import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# gensim
import gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
train_df = pd.read_csv("../input/train.csv")
train_df_insincere = train_df[train_df.target==1]
# select observations that belong to genre of insincere questions
insincere_questions = train_df_insincere['question_text']
# tokenize words and cleaning up punctuations and so
def sent_to_words(insincere_questions):
    for question in insincere_questions:
        yield(gensim.utils.simple_preprocess(str(question), deacc=True))  # deacc=True removes punctuations

question_words = list(sent_to_words(insincere_questions))
# remove stopwords
stop_words = set(stopwords.words("english"))
questions_without_stopwords = [[word for word in simple_preprocess(str(question)) 
                                if word not in stop_words] for question in question_words]

# Form Bigrams
bigram = Phrases(questions_without_stopwords, min_count=5, threshold=100) 
# higher value of the params min_count and threshold will result in less bigrams. 
# You can playaround with these parameters to get better bigrams

bigram_mod = Phraser(bigram)
bigrams = [bigram_mod[question] for question in questions_without_stopwords]
# Create Dictionary of words - This creates id for each word/ phrase
id2word = Dictionary(bigrams) 
print("Word at 0th id: ", id2word[0])

# create corpus - Convert a list of words into the bag-of-words forma
corpus = [id2word.doc2bow(text) for text in bigrams]
print("First element of corpus: ", corpus[0])
# this is bit computation intensive and may take time
coherence_values = []
model_list = []
range_num_topics = range(2,10)
for num_topics in range_num_topics:
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                            random_state=100, update_every=1,
                                            chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    model_list.append(model)
    coherencemodel = CoherenceModel(model=model, texts=bigrams, dictionary=id2word, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())
# Print the coherence scores
for num_topic, coherence_value in zip(range_num_topics, coherence_values):
    print("Number of Topics : ", num_topic, " . Coherence Value: ", round(coherence_value, 3))
# Plot coherence score graph
plt.plot(range_num_topics, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend("coherence_values", loc='best')
plt.show()
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_list[1], corpus, id2word)
vis