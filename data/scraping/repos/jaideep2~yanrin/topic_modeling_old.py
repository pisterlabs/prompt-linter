import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, TfidfModel
import operator
import psycopg2
from utils.helper import get_datequery
from sql.statements import sql_statement
import sys
date = sys.argv[1]
doc2id_mapping = {}
def get_doc(date):
    doc = []
    try:
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        #cur.execute('''select content from news where id>0 and id<50000;''')
        #cur.execute('''select content from news order by random() LIMIT 50000;''')
        cur.execute('''select id,content from news where '''+get_datequery(date)) #5k
        rows = cur.fetchall()
        for i,row in enumerate(rows):
            doc2id_mapping[i] = row[0]
            doc.append(row[1])
        conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return doc

doc = get_doc(date)
print('doc len:',len(doc))
def build_texts(doc):
    for d in doc: yield gensim.utils.simple_preprocess(d, deacc=True, min_len=3)

train_texts = list(build_texts(doc))
print('train len:',len(train_texts))

bigram = gensim.models.Phrases(train_texts, min_count=10)  # for bigram collocation detection
# bigram = gensim.models.phrases.Phraser()
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
print('bigramed train_texts',len(train_texts))
dictionary = Dictionary(train_texts)
print('dict size:',len(dictionary))
#remove extremes
dictionary.filter_extremes(no_below=10, no_above=0.1) #remove words in less than 5 documents and more than 50% documents
dictionary.filter_n_most_frequent(2000) #Filter out 1000 most common tokens
#filter_tokens(bad_ids=None, good_ids=None)
corpus = [dictionary.doc2bow(text) for text in train_texts]
print('corpus size:',len(corpus))
coherences = []
#LSI
'''
lsimodel = LsiModel(corpus=corpus, num_topics=1, id2word=dictionary)
#print(lsimodel.show_topics(num_topics=5)) # Showing only the top 5 topics
lsitopics = lsimodel.show_topics(formatted=False)
lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]
lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
print('LSI:',lsi_coherence)
coherences.append(lsi_coherence)
#LDA
ldamodel = LdaModel(corpus=corpus, num_topics=1, id2word=dictionary)
#print(ldamodel.show_topics(num_topics=5))
ldatopics = ldamodel.show_topics(formatted=False)
ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
lda_coherence = CoherenceModel(topics=ldatopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
print('LDA:',lda_coherence)
coherences.append(lda_coherence)
#HDP
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdptopics = hdpmodel.show_topics(formatted=False)
hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
print('HDP:',hdp_coherence)
coherences.append(hdp_coherence)
'''

def ret_top_model(corpus):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed.

    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    rounds = 1
    high = 0.0
    out_lm = None
    #while top_topics[0][1] < 0.97 and rounds < 2: #0.97
    while True:
        lm = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary, minimum_probability=0)
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

#print('first Tfidf')
#tfidf = TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]
#corpus_tfidf = None
print('then lda+lsi model')
lm, top_topics, high = ret_top_model(corpus)
#lm = LdaModel.load('/tmp/model.ldamodel')
print('finished')
#pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])
#lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]
#lda_lsi_coherence = CoherenceModel(topics=lda_lsi_topics[:5], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
#print('LDA+LSI:',lda_lsi_coherence)
#coherences.append(lda_lsi_coherence)

def put_doc(news_ids,top_words):
    try:
        print('1')
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        type = 'All'
        startdate = date
        enddate = date
        print('2')
        for name in top_words:
            print('3')
            cur.execute(sql_statement,{'name':name,'type':type,'startdate':startdate,'enddate':enddate})
        print('4')
        conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

random_document = doc[80]
print(random_document)
#Process
#train_texts = list(build_texts(doc))
#train_texts = process_texts(train_texts)
#print(train_texts)

print('Lets see some topic labels')
#from topic_labels import topic_labels
#topic_vec = dictionary.doc2bow(topic_labels)

#topic_model = lm[topic_vec]
#topic_model


random_vec = dictionary.doc2bow(random_document.lower().split())
matched = lm[random_vec]
matched = sorted(matched,key=lambda x: float(x[1]),reverse=True)
print('matched:',matched)
top_words = []
print('topics:')
for topicid,_ in matched:
    topic = lm.show_topic(topicid, topn=5)
    print(topic)
    '''for word,freq in topic:
        top_words.append((word,freq))'''
top_words = sorted(top_words,key=lambda x: float(x[1]),reverse=True)
print(top_words[:5])
top_words = [t[0] for t in top_words]
#news_ids = [1,2,3,4,5,6,7,8,9]
#put_doc(doc,top_words)