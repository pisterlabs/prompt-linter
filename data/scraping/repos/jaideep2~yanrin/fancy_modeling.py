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
import pandas as pd
date = sys.argv[1]
doc2id_mapping = {}

def get_doc():
    doc = []
    try:
        conn = psycopg2.connect("dbname=jaideepsingh user=jaideepsingh")
        cur = conn.cursor()
        #cur.execute('''select content from news where id>0 and id<50000;''')
        #cur.execute('''select content from news order by random() LIMIT 50000;''')
        #cur.execute('''select id,content from news where date >= '2016-01-01' and date <= '2016-12-31';''') #5k
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

def process_texts(texts,bigram,stops):
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

def main():
    doc = get_doc()
    print('doc len:', len(doc))

    train_texts = list(build_texts(doc))
    print('train len:', len(train_texts))

    bigram = gensim.models.Phrases(train_texts, min_count=10)  # for bigram collocation detection
    stops = set(stopwords.words('english'))  # nltk stopwords list

    train_texts = process_texts(train_texts, bigram, stops)
    print('bigramed train_texts', len(train_texts))
    vocabulary = Dictionary(train_texts)
    print('vocab size:', len(vocabulary))
    # remove extremes
    vocabulary.filter_extremes(no_below=3, no_above=0.3)  # remove words in less than 5 documents and more than 50% documents
    #vocabulary.filter_n_most_frequent(50)  # Filter out 1000 most common tokens
    # filter_tokens(bad_ids=None, good_ids=None)
    corpus = [vocabulary.doc2bow(text) for text in train_texts]
    print('corpus size:', len(corpus))
    lda = LdaModel(corpus=corpus, id2word=vocabulary, num_topics=10, chunksize=1500, iterations=200,alpha='auto')
    print(pd.DataFrame([[word for rank, (word, prob) in enumerate(words)] for topic_id, words in lda.show_topics(formatted=False, num_words=6, num_topics=35)]))
if __name__ == '__main__':
    main()