import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, TfidfModel
import operator
import psycopg2
from sql.statements import create_new_record, update_news_id, find_record
from utils.helper import get_datequery
doc2id_mapping = {}


def get_doc(date):
    doc = []
    try:
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        cur.execute('''select id,content from news where ''' + get_datequery(date))
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

def build_texts(doc):
    for d in doc: yield gensim.utils.simple_preprocess(d, deacc=True, min_len=3)

def process_doc(doc):
    train_texts = list(build_texts(doc))
    print('train len:', len(train_texts))
    bigram = gensim.models.Phrases(train_texts, min_count=10)  # for bigram collocation detection
    stops = set(stopwords.words('english'))  # nltk stopwords list
    #Stopword Removal.
    texts = [[word for word in line if word not in stops] for line in train_texts]
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

def ret_top_model(corpus, dictionary, train_texts, num_times):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until a certian threshold is crossed.

    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    rounds = 1
    high = 0.0
    out_lm = None
    print('dict size:',len(dictionary))
    num_topics = int(len(dictionary)*0.1) #10% of all dictionary
    print('num_topics:',num_topics)
    while True:
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, minimum_probability=0)
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
        if rounds > num_times-1:
            break
        rounds+=1
    return out_lm, top_topics, high


def get_doc_topics(doc,lm,dictionary,doc_id):
    print('getting topics for doc',doc_id)
    topic_distribution = lm.get_document_topics(dictionary.doc2bow(doc[doc_id].split()), minimum_probability=0.6)
    topic_distribution = sorted(topic_distribution, key=lambda x: float(x[1]), reverse=True)
    topics = []
    for t, f in topic_distribution:
        print('Prob:',t,f)
        for word, prob in lm.show_topic(t):
            topics.append(word)
    return topics

def create_dates(year):
    from datetime import date,timedelta
    start = date(year,1,1)
    dates = []
    while start.year == year and start.month == 1:
        start += timedelta(days=1)
        dates.append(start)
    return dates

def process_dict(train_texts, doc_len):
    dictionary = Dictionary(train_texts)
    print('dict size:', len(dictionary))
    # remove extremes
    no_below = int(doc_len * 0.008)
    filter_freq = int(doc_len * 0.2)
    print('no_below,filter_freq:', no_below, filter_freq)
    dictionary.filter_extremes(no_below=no_below)  # remove words in less 0.8% of documents
    dictionary.filter_n_most_frequent(filter_freq)  # Filter out 20% of most common word tokens
    # filter_tokens(bad_ids=None, good_ids=None)
    return dictionary

def insert_new_row(cur, topic_name, date, news_id):
    cur.execute(create_new_record,
                {'name': topic_name, 'type': 'All', 'startdate': date, 'enddate': date, 'news_ids': '{%s}' % news_id})
    print('new row done')


def update_row(cur, topic_name, date, news_id, row_id):
    cur.execute(update_news_id, {'news_ids': '%s' % news_id, 'id' : row_id})
    print('rows updated')


def find_relevant_rows(cur, topic_name, date):
    cur.execute(find_record, {'name': topic_name, 'date': date})
    return cur.fetchall()


def insert_into_relevant_topic(cur, topic_name, news_id, date):
    rows = find_relevant_rows(cur, topic_name, date)
    print('insert_into_relevant_topic rows:',rows)
    if not rows:
        # if topic name doesnt exist for current date insert new topic with news_ids = 0 and given date
        print('insert_new_row')
        insert_new_row(cur, topic_name, date, news_id)
    else:
        # if topic exists append news_id to said topic
        print('update_row')
        update_row(cur, topic_name, date, news_id, rows[0][0])


def main():
    '''
    0. decide what date or range of dates to run this on
    1. create dictionary and corpus
    2. create model
    3. for each doc get top topic
    4. insert topic into topic table with date
    :return:
    '''
    datez = create_dates(2016)
    doc = []
    for date in datez:
        doc.extend(get_doc(date))
    doc_len = len(doc)
    train_texts = process_doc(doc)
    dictionary = process_dict(train_texts,doc_len)
    corpus = [dictionary.doc2bow(text) for text in train_texts]
    print('doc_len:',doc_len)
    print('corpus size:',len(corpus))
    print('lda+lsi model start!')
    num_times = 1
    lm, top_topics, high = ret_top_model(corpus,dictionary,train_texts,num_times)
    save_model = True
    load_model = False
    if save_model:
        lm.save('/Users/jaideepsingh/Projects/yanrin/lm2016Jan.ldamodel')
    if load_model:
        lm = LdaModel.load('/Users/jaideepsingh/Projects/yanrin/lm2016Jan.ldamodel')
    print('finished!')
    try:
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        for doc_id in range(doc_len):
            topics = get_doc_topics(doc, lm, dictionary, doc_id)
            print(topics)
            print('docid:',doc_id)
            print('mapping:',doc2id_mapping[doc_id])
            news_id = doc2id_mapping[doc_id]
            print('news_id:',news_id)
            if topics:
                for topic_name in topics:
                    print('Putting topic',topic_name,'into topics table')
                    insert_into_relevant_topic(cur, topic_name, news_id, date)
            conn.commit()
        cur.close()
        conn.close()
        print('db closed')
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    main()