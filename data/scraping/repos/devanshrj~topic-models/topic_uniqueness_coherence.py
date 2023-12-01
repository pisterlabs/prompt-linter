from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

import argparse
import pandas as pd
import sqlalchemy
import warnings

warnings.filterwarnings('ignore')


def topic_uniqueness(lexicon, L, renorm=False):
    '''
        lexicon (str): name of topic log-likelihood lexicon to compute uniqueness for
        L (int): number of top words to consider
        renorm (boolean): renormalize topic uniqueness to between 0 and 1, 
                          otherwise topic uniqueness is between 1 / len(topics_dict) and 1
    '''

    print("- Reading topic lexicon -")
    query = f'''SELECT * FROM {lexicon};'''
    lexicon_df = pd.read_sql(query, engine)

    # topics_dict: topic-id -> word -> weight
    print("- Generating topics dict -")
    topics_dict = dict()
    for index, row in lexicon_df.iterrows():
        topic_id = row['category']
        term = row['term']
        weight = row['weight']
        if topic_id in topics_dict:
            topics_dict[topic_id][term] = weight
        else:
            topics_dict[topic_id] = dict()
            topics_dict[topic_id][term] = weight

    K = len(topics_dict)
    if L == 0 or K == 0:
        print("Both L and K must be non-zero")
        return None, None
    
    # sort the dict
    sorted_topics_dict = dict()
    for topic, words in topics_dict.items():
        sorted_words = [k for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)][0:L]
        sorted_topics_dict[topic] = {word: words[word] for word in sorted_words}
    
    print("- Computing TU -")
    topic_scores = dict()
    for topic, words in sorted_topics_dict.items():
        this_score = 0
        for word in words:
            cnt_l_k = 0
            for topic_inner, words_inner in sorted_topics_dict.items():
                if word in words_inner:
                    cnt_l_k += 1
            this_score += 1 / float(cnt_l_k)
        topic_scores[topic] = this_score/float(L)

    TU = sum([v for k,v  in topic_scores.items()]) / float(K)
    if renorm:
        print("- Renorming TU -")
        TU = (TU - 1 / float(K)) / float((1 - 1 / float(K)))
    return TU


def topic_coherence(msg_table, topic_table):
    '''
        msg_table (str): name of message table used to estimate topics
        topic_table (str): name of table with representative words of topics
    '''

    print("- Reading corpus -")
    corpus_query = f'''SELECT * from {msg_table} limit 10000;'''
    corpus_df = pd.read_sql(corpus_query, engine)
    messages_li = corpus_df['message'].tolist()

    print("- Tokenizing corpus -")
    texts = []
    for msg in messages_li:
        if msg is None:
            msg = ''
        texts.append(msg.split())

    print("- Reading topics -")
    topics_query = f'''SELECT * from {topic_table};'''
    topics_df = pd.read_sql(topics_query, engine)
    topics_li = topics_df['termy'].tolist()

    topics = []
    for topic in topics_li:
        topics.append(topic.split(', '))

    dictionary = Dictionary(texts)

    print("- Computing u_mass -")
    cm1 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='u_mass')
    u_mass = cm1.get_coherence()

    print("- Computing c_v -")
    cm2 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    c_v = cm2.get_coherence()

    print("- Computing c_uci -")
    cm3 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_uci')
    c_uci = cm3.get_coherence()

    print("- Computing c_npmi -")
    cm4 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_npmi')
    c_npmi = cm4.get_coherence()

    return u_mass, c_v, c_uci, c_npmi


parser = argparse.ArgumentParser()
parser.add_argument("--database", type=str)
parser.add_argument("--msgs", type=str)
parser.add_argument("--topics", type=str)
parser.add_argument("--lexicon", type=str)
parser.add_argument("--l", type=int)
parser.add_argument("--renorm", default=False, type=bool)
args = parser.parse_args()
print(args)

db = sqlalchemy.engine.url.URL(drivername='mysql',
                               host='127.0.0.1',
                               database=args.database,
                               query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})


engine = sqlalchemy.create_engine(db)

print("----- Calculating Topic Coherence -----")
u_mass, c_v, c_uci, c_npmi= topic_coherence(msg_table=args.msgs, topic_table=args.topics)

print("----- Calculating Topic Uniqueness -----")
TU = topic_uniqueness(lexicon=args.lexicon, L=args.l, renorm=args.renorm)

print("Topic Coherence scores:")
print(f"u_mass = {u_mass}")
print(f"c_v = {c_v}")
print(f"c_uci = {c_uci}")
print(f"c_npmi = {c_npmi}")

print(f"Topic Uniqueness (TU) = {TU}")