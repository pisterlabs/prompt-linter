from operator import index
import pyLDAvis
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

documents = dataset.data

print(len(documents))
print(documents[3])

# 11314
#[0] Well i'm not sure about the story nad it did seem biased. What
# I disagree with is your statement that the U.S. Media is out to
# ruin Israels reputation. That is rediculous. The U.S. media is
# the most pro-israeli media in the world. Having lived in Europe
# I realize that incidences such as the one described in the
# letter have occured. The U.S. media as a whole seem to try to
# ignore them. The U.S. is subsidizing Israels existance and the
# Europeans are not (at least not to the same degree). So I think
# that might be a reason they report more clearly on the
# atrocities.
#         What is a shame is that in Austria, daily reports of
# the inhuman acts commited by Israeli soldiers and the blessing
# received from the Government makes some of the Holocaust guilt
# go away. After all, look how the Jews are treating other races
# when they got power. It is unfortunate.

# [3]
# Notwithstanding all the legitimate fuss about this proposal, how much
# of a change is it?  ATT's last product in this area (a) was priced over
# $1000, as I suspect 'clipper' phones will be; (b) came to the customer
# with the key automatically preregistered with government authorities. Thus,
# aside from attempting to further legitimize and solidify the fed's posture,
# Clipper seems to be "more of the same", rather than a new direction.
#    Yes, technology will eventually drive the cost down and thereby promote
# more widespread use- but at present, the man on the street is not going
# to purchase a $1000 crypto telephone, especially when the guy on the other
# end probably doesn't have one anyway.  Am I missing something?
#    The real question is what the gov will do in a year or two when air-
# tight voice privacy on a phone line is as close as your nearest pc.  That
# has got to a problematic scenario for them, even if the extent of usage
# never surpasses the 'underground' stature of PGP.

import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import preprocess_string

nltk.download('stopwords')


def clean_text(d):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', d)
    return text

def clean_stopword(d):
    stop_words = stopwords.words('english')
    return ' '.join([w.lower() for w in d.split() if w.lower() not in stop_words and len(w) > 3])

def preprocessing(d):
    return preprocess_string(d)

import pandas as pd

news_df = pd.DataFrame({'article':documents})
# print(len(news_df))
# 11314

news_df.replace("", float("NaN"), inplace=True)
# print(news_df.isnull().values.any())
# True

news_df.replace("", float("NaN"), inplace=True)
news_df.dropna(inplace=True)
# print(len(news_df))

# 11096

news_df['article'] = news_df['article'].apply(clean_text)
print(news_df['article'])
# 0        Well i'm not sure about the story nad it did s...
# 1        \n\n\n\n\n\n\nYeah, do you expect people to re...
# 2        Although I realize that principle is not one o...
# 3        Notwithstanding all the legitimate fuss about ...
# 4        Well, I will have to change the scoring on my ...
#                                ...
# 11309    Danny Rubenstein, an Israeli journalist, will ...
# 11310                                                   \n
# 11311    \nI agree.  Home runs off Clemens are always m...
# 11312    I used HP DeskJet with Orange Micros Grappler ...
# 11313                                          ^^^^^^\n...
# Name: article, Length: 11096, dtype: object

news_df['article'] = news_df['article'].apply(clean_stopword)
print(news_df['article'])

# 0        well sure story seem biased. what disagree sta...
# 1        yeah, expect people read faq, etc. actually ac...
# 2        although realize principle strongest points, w...
# 3        notwithstanding legitimate fuss proposal, much...
# 4        well, change scoring playoff pool. unfortunate...
#                                ...
# 11309    danny rubenstein, israeli journalist, speaking...
# 11310
# 11311    agree. home runs clemens always memorable. kin...
# 11312    used deskjet orange micros grappler system6.0....
# 11313    ^^^^^^ argument murphy. scared hell came last ...
# Name: article, Length: 11096, dtype: object

tokenized_news = news_df['article'].apply(preprocessing)
tokenized_news = tokenized_news.to_list()
# print(tokenized_news)

import numpy as np

drop_news = [index for index, sentence in enumerate(tokenized_news) if len(sentence) <= 1]
news_texts = np.delete(tokenized_news, drop_news, axis=0)
# print(len(news_texts))

# 10936

from gensim import corpora

dictionary = corpora.Dictionary(news_texts)
corpus = [dictionary.doc2bow(text) for text in news_texts]

# print(corpus[1])

# [(51, 1), (52, 1), (53, 1), (54, 1), (55, 1), (56, 1), (57, 1), (58, 2), (59, 1), (60, 1), (61, 1), (62, 2), (63, 1), (64, 1), (65, 1), (66, 1), (67, 1),
#  (68, 2), (69, 3), (70, 1), (71, 1), (72, 1), (73, 1), (74, 1), (75, 2), (76, 1), (77, 1), (78, 1), (79, 1), (80, 1), (81, 1), (82, 2), (83, 1), (84, 1), (85, 1), (86, 1)]

from gensim.models import LsiModel

# lsi_model = LsiModel(corpus, num_topics=20, id2word=dictionary)
# topics = lsi_model.print_topics()
# print(topics)

from gensim.models.coherencemodel import CoherenceModel

# min_topics, max_topics = 20, 25
# coherence_scores = []

# for num_topics in range(min_topics, max_topics):
#     model = LsiModel(corpus, num_topics=num_topics, id2word=dictionary)
#     coherence = CoherenceModel(model=model, texts=news_texts, dictionary=dictionary)
#     coherence_scores.append(coherence.get_coherence())
    
# print(coherence_scores)

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')

# x=[int(i) for i in range(min_topics, max_topics)]

# plt.figure(figsize=(10,6))
# plt.plot(x, coherence_scores)
# plt.xlabel('Number of Topics')
# plt.ylabel('Coherence Scores')
# plt.show()

# lsi_model = LsiModel(corpus, num_topics=24, id2word=dictionary)
# topics = lsi_model.print_topic(num_topics=24)
# print(topics)

from gensim.models import LdaModel

# lda_model = LdaModel(corpus, num_topics=20, id2word=dictionary)
# topics = lda_model.print_topics()
# print(topics)

from gensim.models.coherencemodel import CoherenceModel

# min_topics, max_topics = 20, 25
# coherence_scores = []

# for num_topics in range(min_topics, max_topics):
#     model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
#     coherence = CoherenceModel(model=model, texts=news_texts, dictionary=dictionary)
#     coherence_scores.append(coherence.get_coherence())

# print(coherence_scores)    

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')

# x=[int(i) for i in range(min_topics, max_topics)]

# plt.figure(figsize=(10,6))
# plt.plot(x, coherence_scores)
# plt.xlabel('Number of Topics')
# plt.ylabel('Coherence Scores')
# plt.show()

################################################################################################
# lda_model = LdaModel(corpus, num_topics=23, id2word=dictionary)
# topics = lda_model.print_topics(num_topics=23)
# print(topics)

# import pyLDAvis.gensim_models
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
# pyLDAvis.display(vis)   

# pandas pyLDAvis ====> version issue It is not working here need more research
#####################################################################################################

