import os.path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from string import punctuation
from gensim import corpora
import pandas as pd
nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def obtain_ov_from_meta(df_meta: pd.DataFrame, titles: list):
    df_meta['title'] = df_meta['title'].apply(lambda x: str(x).strip().lower())
    ovs = []
    oov = 0
    for title in titles:
        matched = re.match(r'(?P<part1>.+) \((?P<part2>.+)\)$', title)
        title1 = matched.group('part1') if matched else title
        title2 = matched.group('part2') if matched else ""
        matched = re.match(r'(?P<part1>.+), (?P<part2>The|An|A|the|a|an)$', title1)
        title1 = matched.group('part2') + ' ' + matched.group('part1') if matched else title1
        title1 = title1.strip().lower()
        title2 = title2.strip().lower()
        candidates = df_meta[(df_meta['title'] == title1) | (df_meta['title'] == title2)]['overview'].values
        if len(candidates) > 0:
            ovs.append(str(candidates[0]))
        else:
            ovs.append("")
            oov += 1
    print('the oov number is {0}'.format(oov))
    return ovs


def split_year_from_title(titles):
    years, topics = [], []
    for title in titles:
        matched = re.match(r'(?P<topic>.+)\((?P<year>\d{4})\)$', title)
        year = matched.group('year') if matched else None
        topic = matched.group('topic') if matched else None
        topic = str(topic).strip()
        years.append(int(year))
        topics.append(topic)
    return years, topics


def get_topic_from_LDA(ov: pd.Series, model_dir, n_topics, train=False):
    model_path = os.path.join(model_dir, f'ldamodel_{n_topics}')
    dict_path = os.path.join(model_dir, f'ldadict_{n_topics}')
    info = ov.values
    info_tokens = [
        lemmatization(remove_stopwords(remove_punct(decontraction(remove_escape_sequences(str(item)).strip()))))
        for item in info]
    if train:
        dic, model, ch = train_LDA(info_tokens, n_topics, model_path, dict_path)
        print(f'coherence value is {ch}')
    else:
        if os.path.exists(model_path) and os.path.exists(dict_path):
            model = LdaModel.load(model_path)
            dic = Dictionary.load(dict_path)
            cm = CoherenceModel(model=model, texts=info_tokens, coherence='c_v', dictionary=dic)
            ch = cm.get_coherence()
            print(f'coherence value is {ch}')
        else:
            raise ValueError(f'LDA model for {n_topics} topics can not be found')

    topic_weights = predict_LDA(model, dic, info_tokens)
    weights = reform_topic_weights(topic_weights, n_topics)

    return weights, ch


def reform_topic_weights(topic_weights, n_topic):
    formed_weights = []
    for weight_list in topic_weights:
        tmp = [0.] * n_topic
        for weight in weight_list:
            tmp[weight[0]] = weight[1]
        formed_weights.append(tmp)
    return formed_weights


def train_LDA(list_sents, n_topics, model_path, dict_path):
    dic = corpora.Dictionary()
    bow_corpus = [dic.doc2bow(doc, allow_update=True) for doc in list_sents]
    model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, num_topics=n_topics,
                                            id2word=dic, random_state=2022,
                                            chunksize=64, passes=10, alpha='auto', per_word_topics=True)
    cm = CoherenceModel(model=model, texts=list_sents, coherence='c_v', dictionary=dic)
    model.save(model_path)
    dic.save(dict_path)
    ch = cm.get_coherence()
    return dic, model, ch


def predict_LDA(model, dic, sents):
    bow_corpus = [dic.doc2bow(sent, allow_update=True) for sent in sents]
    topic_weights = [model.get_document_topics(sent) for sent in bow_corpus]
    return topic_weights


def remove_stopwords(text):
    stop = stopwords.words('english')
    return [word.lower() for word in word_tokenize(text) if word.lower() not in stop]


def remove_escape_sequences(text):
    return text.translate(str.maketrans("\n\t\r", "   "))


def decontraction(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def remove_punct(text):
    table = str.maketrans('', '', punctuation)
    return text.translate(table)


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return [stemmer.stem(lemmatizer.lemmatize(word, pos='n')) for word in text]


if __name__ == '__main__':
    folder_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(folder_dir, 'ml-1m')
    # item and meta data
    item_header = ['item_id', 'title', 'genres']
    df_item = pd.read_csv(os.path.join(data_dir, 'movies.dat'), sep='::', names=item_header)
    df_item.drop(columns=['title'], axis=1, inplace=True)
    # meta data
    meta_header = ['movieId', 'title', 'release_date', 'rating', 'overview']
    df_meta = pd.read_csv(os.path.join(data_dir, 'movie_data.csv'))[meta_header]
    df_meta.rename(columns={'movieId': 'item_id', 'rating': 'avg_score'}, inplace=True)
    df_item = df_item.merge(df_meta, on='item_id', how='left')
    # behavior
    bhv_header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_bhv = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::', names=bhv_header)
    selected_movies = df_bhv['item_id'].value_counts(sort=True, ascending=False).index.tolist()[0:1000]
    df_item = df_item[df_item['item_id'].apply(lambda x: x in selected_movies)].reset_index(drop=True)

    df_item['title'] = df_item['title'].apply(lambda x: str(x) + ' ')
    df_item['info'] = df_item['title'] + df_item['overview']
    movie_info = df_item['info']

    best_ch = -1
    best_num = 0
    for n_topic in range(4, 24, 2):
        print('{0} topics is being done'.format(n_topic))
        weights, ch = get_topic_from_LDA(ov=movie_info,
                                         model_dir=os.path.join(folder_dir, 'saved_models'),
                                         n_topics=n_topic,
                                         train=True)
        if ch > best_ch:
            best_ch, best_num = ch, n_topic
        print('{0} topics is done'.format(n_topic))
        print('{0} topics cv is {1}'.format(n_topic, ch))

    print('The best number of topics for LDA is {0}'.format(best_num))
