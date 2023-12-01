import json
import gensim as gensim
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt

# reading post details from json file
f = open('articles.json', )
data = json.load(f)
posts = []
# print(data['preprocessed']['0'])
for i in data['preprocessed']:
    posts.append(data['preprocessed'][i])

# creating a bag of unique words in all the posts
dictionary = gensim.corpora.Dictionary(posts)
# print(len(dictionary))
# removing frequent and infrequent words
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)
# print(len(dictionary))
# consists of frequency count of each word occuring in an article
bow_corpus = [dictionary.doc2bow(doc) for doc in posts]
# print(bow_corpus)

perplexity = []
bayesian = []
coherence = []
model = []
topics = []
for i in range(10, 31, 2):
    topics.append(i)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=i, id2word=dictionary, passes=2, workers=2)
    model.append(lda_model)
    perplexity.append(lda_model.log_perplexity(bow_corpus))
    coherence.append(
        CoherenceModel(model=lda_model, texts=posts, dictionary=dictionary, coherence='c_v').get_coherence())

# print(topics)
# print(coherence)
# print(perplexity)

plt.figure();
plt.plot(topics, coherence)
plt.title('Measurement:Coherence')
plt.xlabel('Number of topics')
plt.xticks(topics)
plt.ylabel('Coherence')
plt.show()

plt.figure();
plt.plot(topics, perplexity)
plt.title('Measurement:Perplexity')
plt.xlabel('Number of topics')
plt.xticks(topics)
plt.ylabel('Perplexity')
plt.show()

# 20 words per topic
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=14, id2word=dictionary, passes=2, workers=2)
words = {}
for i in range(10):
    w = lda_model.get_topic_terms(i, topn=20)
    words[i] = [dictionary[pair[0]] for pair in w]

print(words)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

topic_list = []
for article in bow_corpus:
    topic = lda_model.get_document_topics(article)
    # print('before', topic)
    topic.sort(key=lambda x: x[1])
    # print('after', topic)
    topic_list.append(topic[-1])

# print(topic_list)
