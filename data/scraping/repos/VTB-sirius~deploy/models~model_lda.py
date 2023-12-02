# -*- coding: utf-8 -*-
import os
import sys
import boto3
import nltk # стоп-слова, пунктуация
import gensim # препроцессинг и модели
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
import razdel
import warnings
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm import tqdm
from pprint import pprint
from gensim.models import CoherenceModel
import pandas as pd
import json
import numpy as np
from sklearn.manifold import TSNE
from pymongo import MongoClient
from urllib.parse import quote_plus as quote
from bson.objectid import ObjectId

nltk.data.path.append('/app/nltk_data')
warnings.filterwarnings("ignore")

def preprocessing(data):
    def tokenize_with_razdel(text):
        return [token.text for token in razdel.tokenize(text)]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    data = [str(i) for i in data]
    data = [i.lower() for i in data]
    text = [i.replace('\xa0', ' ') for i in data]
    sentences = [nltk.sent_tokenize(paragraph) for paragraph in text] # токенизация по предложениям
    sentences = [sen for sublist in sentences for sen in sublist]
    tokenized_sentences = [tokenize_with_razdel(sentence) for sentence in sentences]
    clean_tokenized_sentences = [
        [word for word in sentence if word.isalpha()] 
        for sentence in tokenized_sentences]
    stop_words = stopwords.words('russian') + ['это', 'который', 'также', 'например']
    clean_tokenized_sentences = [
        [token for token in sent if token not in stop_words and len(token) > 1] 
        for sent in clean_tokenized_sentences]

    mystem = Mystem() # лемматизатор
    clean_tokenized_sentences = [
        [lemma for lemma in mystem.lemmatize(' '.join(token_sentence)) if not lemma.isspace()] 
        for token_sentence in tqdm(clean_tokenized_sentences, desc='Lemmatization progress')]
    clean_tokenized_sentences = [
        [token for token in sent if token not in stop_words and len(token) > 1] 
        for sent in clean_tokenized_sentences]
    # Строим модели биграм и триграм
    bigram = gensim.models.Phrases(
        clean_tokenized_sentences,
        min_count=5, 
        threshold=100)  # ,больше threshold == меньше фраз.
    trigram = gensim.models.Phrases(bigram[clean_tokenized_sentences], threshold=100)

    # Более быстрый способ превратить фразу в триграмму/биграмму
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    clean_tokenized_sentences = make_trigrams(clean_tokenized_sentences)
    id2word = corpora.Dictionary(clean_tokenized_sentences)
    id2word.filter_extremes(no_below=10, no_above=0.7)  #изучи параметр
    # Create Corpus
    texts = clean_tokenized_sentences
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    print('Preprocessing finished successfully!')
    return corpus, id2word, sentences

def format_topics_sentences(ldamodel, corpus, texts):
    topic_num_l, prop_topic_l, topic_keywords_l, row_l = [], [], [], []
    sent_topics_df = pd.DataFrame()
    for i, row in tqdm(enumerate(ldamodel[corpus]), desc='Building of table', total=len(corpus)):
        elem = max(row, key=lambda x: x[1])
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        topic_num, prop_topic = elem
        wp = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, _ in wp])
        topic_num_l.append(int(topic_num))
        prop_topic_l.append(round(prop_topic, 4))
        topic_keywords_l.append(topic_keywords)
        row_l.append(row)
    tab = pd.DataFrame(topic_num_l, columns=['topic_num'])
    tab['prop_topic'] = prop_topic_l
    tab['topic_keywords'] = topic_keywords_l
    tab['row'] = row_l
    contents = pd.Series(texts)
    tab = pd.concat([tab, contents], axis=1)
    return tab

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇 alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class LDAMalletModel():
    def __init__(self):
        print('...LDA Mallet...')
        self.send = {}

    def fit(self, data, mallet_path='/app/mallet-2.0.8/bin/mallet', 
            preproc_func=preprocessing, num_topics=10, prefix=''):
        self.corpus, id2word, self.data = preproc_func(data)
        self.ldamallet = LdaMallet(mallet_path=mallet_path,
                                   corpus=self.corpus,
                                   num_topics=num_topics,
                                   id2word=id2word,
                                   prefix=prefix,
                                   workers=8)
        print('Model training finished successfully!')

    def show_table(self):
        df_topic_sents_keywords = format_topics_sentences(
            ldamodel=self.ldamallet, corpus=self.corpus, texts=self.data)
        print('Building of table finished successfully!')
        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 
            'Keywords', 'Prob_of_all_topics', 'Text']
        if 'cord_x' not in df_dominant_topic.columns:
            sp = []
            for i in tqdm(df_dominant_topic['Prob_of_all_topics'].tolist(), 
                          desc='Collecting all probabilities'):
                tmp = []
                for j in i:
                    tmp.append(j[1])
                sp.append(tmp)
            X = np.array(sp)
            print('TSNE started...')
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

            x = X_embedded[:, 0]
            y = X_embedded[:, 1]
            df_dominant_topic = pd.concat([df_dominant_topic, pd.Series(x)], axis=1)
            df_dominant_topic = pd.concat([df_dominant_topic, pd.Series(y)], axis=1)
            df_dominant_topic.columns = [
                'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 
                'Keywords', 'Prob_of_all_topics', 'Text', 'cord_x', 'cord_y']
        # Show
        return df_dominant_topic

    def intertopic_map(self):
        return sorted(self.ldamallet.show_topics(formatted=False, num_topics=-1), key=lambda x: x[0])

    def get_topics_by_doc_id(self, table, doc_id):
        dct = {'content': table.loc[doc_id]['Text']}
        distr = []
        for i in table.loc[doc_id]['Prob_of_all_topics']:
            distr.append({'label': f'Topic {i[0]}', 'value': i[1]})
        dct['distribution'] = distr
        return json.dumps(dct)

    def get_payload(self, table, prj_id):
        dct = {'payload': {'intertopic_map': []}}
        # Собираем intertopic map
        m = self.intertopic_map()
        for i in tqdm(m, desc='Intertopic map'):
            tmp = {'id': i[0]}
            keywords = []
            for j in i[1]:
                keywords.append(j[0])
            tmp['keywords'] = keywords
            tmp['cord_x'] = table[table['Dominant_Topic'] == i[0]]['cord_x'].mean()
            tmp['cord_y'] = table[table['Dominant_Topic'] == i[0]]['cord_y'].mean()
            tmp['size'] = len(table[table['Dominant_Topic'] == i[0]].index)
            dct['payload']['intertopic_map'].append(tmp)
        # Собираем topics
        dct['payload']['topics'] = []
        topics = sorted([i[0] for i in m])
        for i in topics:
            tmp = {
                'id': i, 
                'keywords': table[table['Dominant_Topic'] == i]['Keywords'].tolist()[0].split(', ')
            }
            dct['payload']['topics'].append(tmp)
        # Собираем documents
        dct['payload']['documents'] = []
        for i in tqdm(range(len(self.corpus)), desc='Documents'):
            dct['payload']['documents'].append({
                '_id': prj_id + '_lda_' + str(i + 1),
                'cord_x': table.loc[i]['cord_x'],
                'cord_y': table.loc[i]['cord_y'],
                'cluster_id': table.loc[i]['Dominant_Topic'],
                'description': table.loc[i]['Keywords'].capitalize()})
        return dct

    def tot(self, table, dataset, date_1, date_2):
        dates = []
        unique_dates = [i for i in dataset['День'].tolist() if i not in unique_dates]
        for i in unique_dates:
            pass
        pass # пока не понимаю как это сделать (Topics over Time)

    def get_topics_by_doc_id(self, table, prj_id):
        res = []
        for i in table.index:
            dct = {
                # for every doc -> prj_id + '_lda_' + id = _id
                '_id': prj_id + '_lda_' + str(i + 1),
                'content': table.loc[i]['Text']
            }
            distr = []
            for j in table.loc[i]['Prob_of_all_topics']:
                distr.append({'label': f'Topic {j[0]}', 'value': j[1]})
            top_idx = max(range(len(distr)), key=lambda index: distr[index]['value'])
            top_ds = distr[top_idx]
            dct['distribution'] = distr
            dct['top_cluster'] = top_ds
            res.append(dct)
        return res

# ----- S3 credentials -----
S3_BUCKET = os.environ['S3_BUCKET']
SERVICE_NAME = os.environ['SERVICE_NAME']
KEY = os.environ['KEY']
SECRET = os.environ['SECRET']
ENDPOINT = os.environ['ENDPOINT']
SESSION = boto3.session.Session()
S3_CLT = SESSION.client(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
S3_RES = SESSION.resource(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
# ----- MongoDB credentials -----
MONGODB_HOST = os.environ['MONGODB_HOST']
MONGODB_DATABASE = os.environ['MONGODB_DATABASE']
MONGODB_USERNAME = os.environ['MONGODB_USERNAME']
MONGODB_PASSWORD = os.environ['MONGODB_PASSWORD']

def generate_value(input_data):
    generated_value = None
    try:
        """
        Для работы некоторых функций необходимо сначала сгенерировать таблицу методом show_table(), 
        а затем передовать её в другие методы
        Методы, для которых необходима таблица: get_topics_by_doc_id(table, doc_id), get_payload(table).

        Обращаю внимание, в тех же функциях я возвращаю словарь Pyhton, а не строку в формате json.

        Предвижу ошибки, которые могут произойти: огромный размер json (там действительно много информации), 
        на данный момент я не могу передавать точное описание документа (сейчас я использую ключевые слова 
        темы, к которой относится документ).
        В файле excel нет меток времени для каждого документа, поэтому в функции get_payload() 
        я закомментировал код, отвечающий за формирование timestamps

        get_payload(table) - возвращает строку в формате json с информацией о payload, как в документации к api
        get_topics_by_doc_id(table, doc_id) - возвращает строку в формате json с информацией о распределении тем 
        в документе, как в документации к api

        """
        get_object_response = S3_CLT.get_object(
            Bucket=S3_BUCKET,
            Key=input_data['file_name_input']
        )
        dataset = pd.read_excel(
            get_object_response['Body'].read(),
            engine='openpyxl')
        dataset.drop_duplicates(subset='text', keep='first', inplace=True)
        data = dataset[dataset.index < 10000]['text'].tolist()
        lda = LDAMalletModel()
        lda.fit(data, num_topics=int(input_data['num_clusters']))
        table = lda.show_table()
        result = lda.get_topics_by_doc_id(table, input_data['prj_id'])
        generated_value = {
            'status': 'Success',
            'code': str(200),
            'payload': lda.get_payload(table, input_data['prj_id'])['payload']
        }
        url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
            user=quote(MONGODB_USERNAME),
            pw=quote(MONGODB_PASSWORD),
            hosts=','.join([MONGODB_HOST]),
            rs='rs01',
            auth_src=MONGODB_DATABASE
        )
        db = MongoClient(url, tlsCAFile='/app/CA.pem')['classify']
        filter = { '_id': ObjectId(input_data['prj_id']) } 
        payload = json.loads(json.dumps(generated_value['payload'], cls=NpEncoder))
        newValue = { "$set": { 'lda_payload': payload } }
        db['projects'].update_one(filter, newValue)
        result = json.loads(json.dumps(result, cls=NpEncoder))
        db['documents'].insert_many(result)
    except Exception as e:
        generated_value = {
            'status': 'Server error',
            'code': str(500),
            'error': str(e)
        }
        print('ERROR', e)
    return generated_value

def main():
    input_data = dict(
        num_clusters = int(sys.argv[3]),
        prj_id = sys.argv[1],
        file_name_input = sys.argv[2]
    )
    print(input_data)
    output_data = generate_value(input_data)
    print('LDA finished')

if __name__ == "__main__":
    main()
