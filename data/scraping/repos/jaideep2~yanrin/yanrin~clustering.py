import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, TfidfModel
import operator
import psycopg2
import numpy as np
def get_doc():
    doc = []
    try:
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        #cur.execute('''select content from news where id>0 and id<50000;''')
        #cur.execute('''select content from news order by random() LIMIT 50000;''')
        cur.execute('''select content from news where date='2016-01-06';''') #5k
        rows = cur.fetchall()
        doc = [row[0] for row in rows]
        conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return doc

doc = get_doc()
print(len(doc))
def build_texts(doc):
    for d in doc: yield gensim.utils.simple_preprocess(d, deacc=True, min_len=3)

train_texts = list(build_texts(doc))

bigram = gensim.models.Phrases(train_texts, min_count=10)  # for bigram collocation detection

stops = set(stopwords.words('english'))  # nltk stopwords list

def process_texts(texts):
    #Stopword Removal.
    texts = [[word for word in line if word not in stops] for line in texts]
    #Collocation detection.
    texts = [bigram[line] for line in texts]
    # Remove numbers, but not words that contain numbers.
    texts = [[token for token in line if not token.isnumeric()] for line in texts]
    # Remove words that are only two or less characters.
    texts = [[token for token in line if len(token) > 2] for line in texts]
    #Lemmatization (not stem since stemming can reduce the interpretability).
    lemmatizer = WordNetLemmatizer()

    texts = [[word for word in lemmatizer.lemmatize(' '.join(line), pos='v').split()] for line in texts]
    return texts

train_texts = process_texts(train_texts)

dictionary = Dictionary(train_texts)
print(len(dictionary))
#remove extremes
dictionary.filter_extremes(no_below=2, no_above=0.02) #remove words in less than 5 documents and more than 50% documents
#dictionary.filter_n_most_frequent(1000) #Filter out 1000 most common tokens

corpus = [dictionary.doc2bow(text) for text in train_texts]
lm = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)

def ret_top_model():
    top_topics = [(0, 0)]
    rounds = 1
    high = 0.0
    out_lm = None
    #while top_topics[0][1] < 0.97 and rounds < 2: #0.97
    while True:
        lm = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
        if high < top_topics[0][1]:
            high = top_topics[0][1]
            out_lm = lm
        print('round ',rounds,':',top_topics[0][1])
        if rounds > 2:
            break
        rounds+=1
    return out_lm, top_topics, high

print('lda+lsi model')
#lm, top_topics, high = ret_top_model()
#print('final:',high)

#save and load
#lm.save('/Users/jaideepsingh/Projects/model.ldamodel')
lm = lm.load('/Users/jaideepsingh/Projects/model.ldamodel')
corpus_lm = lm[corpus]
print('loading done')

print('Clustering')
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
from sklearn.cluster import KMeans
from gensim.matutils import corpus2dense
k = 5

print('num_terms:',lm.num_terms)
x_kmeans = corpus2dense(corpus=corpus_lm,num_terms=lm.num_terms)
kmeans = KMeans(n_clusters=k)
kmeans.fit(x_kmeans)
y_kmeans = kmeans.predict(x_kmeans)
labels = kmeans.labels_
print('labels:',len(labels))
centroids = kmeans.cluster_centers_
print('centroids:',len(centroids))
for i in range(k):
    #print('i:',i)
    # select only data observations with cluster label == i
    ds = x_kmeans[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'k,')
    print('i:',i,'ds[:,0]:',ds[:,0],'ds[:,1]:',ds[:,1])
    pyplot.plot()
    # plot the centroids
    #lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    #pyplot.setp(lines)#,ms=15.0)
    #pyplot.setp(lines,mew=2.0)
pyplot.show()