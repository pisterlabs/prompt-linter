# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:25:43 2019
@author: Taufik Sutanto
taufik@tau-data.id
https://tau-data.id

~~Perjanjian Penggunaan Materi & Codes (PPMC) - License:~~
* Modul Python dan gambar-gambar (images) yang digunakan adalah milik dari berbagai sumber sebagaimana yang telah dicantumkan dalam masing-masing license modul, caption atau watermark.
* Materi & Codes diluar point (1) (i.e. "taudata.py" ini & semua slide ".ipynb)) yang digunakan di pelatihan ini dapat digunakan untuk keperluan akademis dan kegiatan non-komersil lainnya.
* Untuk keperluan diluar point (2), maka dibutuhkan izin tertulis dari Taufik Edy Sutanto (selanjutnya disebut sebagai pengarang).
* Materi & Codes tidak boleh dipublikasikan tanpa izin dari pengarang.
* Materi & codes diberikan "as-is", tanpa warranty. Pengarang tidak bertanggung jawab atas penggunaannya diluar kegiatan resmi yang dilaksanakan pengarang.
* Dengan menggunakan materi dan codes ini berarti pengguna telah menyetujui PPMC ini.
"""
import warnings; warnings.simplefilter('ignore')
import tweepy, googlemaps, re, itertools, os, time
from html import unescape
from tqdm import tqdm
from unidecode import unidecode
from nltk import sent_tokenize
from textblob import TextBlob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tag import CRFTagger
import spacy
import numpy as np, pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import networkx as nx, operator
from dateutil import parser
from datetime import datetime
import pickle
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from math import radians, cos, sin, asin, sqrt
from wordcloud import WordCloud


HTfilters = set(['zz', 'architec', 'prize', 'stirli', 'architect', 'london', 'cpd', 'design', 'stirling', 'photogr', 'gemini',
             'mule', 'karaoke', 'playing', 'official', 'berita', 'follow', 'retweet', 'mufc', 'ntms', 'infolimit', 'eeaa',
             'eaa', 'cfc', 'caprico', 'breaking','news', 'libra', 'mereka', 'brankas', 'psikolog', 'aquarius', 'klc'])


file = 'data/provinsi-latlon-radius.csv'
prov = pd.read_csv(file)
Ck = 'WQWIh4BC4dVio7xiV4NrU9Z29' # consumer_key
Cs = 'yhxEHCiJCsjnv4joJakVG25Nqm9EJ8ec1pLqHqpneAtqYdjgLL' # consumer_secret
At = '2214118411-i0MsjqZqjZ6uPfUplghxXcJsXdNYhRsCO82AnPW' # access_token
As = 'hxjsnKSY8dgv4Cl5gQd6M6Oax27U7xVoZrHnvSvRdBlCx' # access_secret
tKeys = (Ck, Cs, At, As)
qry = 'banjir OR gempa OR longsor OR tsunami'
N=300
lan='id'

def getData(qry, N=300, prov=None, lan='id', tKeys=None):
    #MaxIter = int(N/100)
    user_ = {"created_at":[], "screen_name": [], "name":[], "followers_count":[], "friends_count":[], "description":[], "id_str":[],
             "location":[], "lat":[], "lon":[], "protected":[], "statuses_count":[], "profile_image_url_https":[], "verified":[], "gender":[], "age":[]}
    tweet_ = {"created_at":[], "screen_name":[], "tweet":[], "retweet_count":[], "favorite_count":[], "location":[], "lat":[], "lon":[]}
    print("Mengambil sampel data dari:")
    mulai = time.time()
    T = []
    for i, p in prov.iterrows():
        propinsi = p['propinsi']
        print(propinsi, end=', ')
        Geo = ','.join([str(p.lat), str(p.lon), str(p.radius)+'km'])
        api = connect(con="twitter", key=tKeys, verbose=False)
        try:
            T2 = api.search_tweets(q=qry, geocode=Geo, lang=lan, count=N, tweet_mode = 'extended')
            if T2:
                T.extend([t._json for t in T2])
        except Exception as err_:
            print("error in Function getData: \n", err_)
        #break

        for t_ in T:#tweepy.Cursor(api.search_tweets, q=qry, geocode=Geo, lang='id', tweet_mode='extended').items(N):
            wkt_ = parser.parse(t_['user']['created_at'])
            wkt_ = datetime.strftime(wkt_, '%Y-%m-%d %H:%M:%S')
            user_['created_at'].append(wkt_)
            user_['screen_name'].append(t_['user']['screen_name'])
            user_['name'].append(t_['user']['name'])
            user_['followers_count'].append(t_['user']['followers_count'])
            user_['friends_count'].append(t_['user']['friends_count'])
            user_['description'].append(t_['user']['description'])
            user_['id_str'].append(t_['user']['id_str'])
            if t_['user']['location']:
                user_['location'].append(t_['user']['location'])
            else:
                user_['location'].append(propinsi)
            user_['lat'].append(p.lat)
            user_['lon'].append(p.lon)
            user_['protected'].append(t_['user']['protected'])
            user_['statuses_count'].append(t_['user']['statuses_count'])
            user_['profile_image_url_https'].append(t_['user']['profile_image_url_https'])
            user_['verified'].append(t_['user']['verified'])
            user_['gender'].append('')
            user_['age'].append(0)

            wkt_ = parser.parse(t_['created_at'])
            wkt_ = datetime.strftime(wkt_, '%Y-%m-%d %H:%M:%S')
            tweet_['created_at'].append(wkt_)
            tweet_['screen_name'].append(t_['user']['screen_name'])
            tweet_['tweet'].append(t_['full_text'])
            tweet_['retweet_count'].append(t_['retweet_count'])
            tweet_['favorite_count'].append(t_['favorite_count'])
            tweet_['location'].append(propinsi)
            tweet_['lat'].append(p.lat)
            tweet_['lon'].append(p.lon)
    waktu = time.time() - mulai
    print('\n\n Finished Collecting {} samples of data about "{}" from all provinces in Indonesia in {} minutes'.format(len(tweet_['tweet']), qry, int(waktu/60)))
    tweet_ = pd.DataFrame(tweet_)
    tweet_.drop_duplicates(subset=["screen_name", "tweet"], keep="first", inplace=True)
    tweet_.sort_values(by=['retweet_count', 'favorite_count'], ascending=False, inplace=True)
    user_ = pd.DataFrame(user_)
    user_.drop_duplicates(subset=["screen_name"], keep="first", inplace=True)
    user_.sort_values(by=['followers_count'], ascending=False, inplace=True)
    return tweet_, user_

getHashTags = re.compile(r"#(\w+)")
def hashTags(df, N=30):
    HTfilters = set(['zz', 'architec', 'prize', 'stirli', 'architect', 'london', 'cpd', 'design', 'stirling', 'photogr', 'gemini',
                 'mule', 'karaoke', 'playing', 'official', 'berita', 'follow', 'retweet', 'mufc', 'ntms', 'infolimit', 'eeaa',
                 'eaa', 'cfc', 'caprico', 'breaking','news', 'libra', 'mereka', 'brankas', 'psikolog', 'aquarius', 'klc'])
    HT = {'hashtags':[]}
    count = 0

    for i, d in tqdm(df.iterrows()):
        hashtags = re.findall(getHashTags, d.tweet)
        if hashtags:
            TG = []
            for tag in hashtags:
                dTag = str(tag).strip().lower()
                if len(dTag)>2:
                    add = True
                    for f in HTfilters:
                        if f in dTag:
                            add=False; break
                    if add:
                        TG.append('#'+dTag); count += 1
            HT['hashtags'].append(TG)
    dtHT = [x for t in tqdm(HT['hashtags']) for x in t] # any(h not in x for h in HTfilters)
    dtHT = pd.Series(dtHT)
    dtHT = dtHT.value_counts()
    dtHT = dtHT.sort_index()
    dtHT = dtHT.sort_values(ascending = False)
    dtHT.to_csv('data/hashTags_Energy_Satrio.csv', encoding='utf8')
    dtHT = dtHT.iloc[:N]
    topHT = [t.lower().strip().replace('#','') for t in dtHT.index]
    print('Plot "{}" HashTag terbanyak'.format(N))
    _ = dtHT.plot(kind='barh', figsize=(12,8), legend = False)
    return topHT

def heatmap(df):
    lat, lon, hashTags, tweets = [], [], [], []
    for i, r in tqdm(df.iterrows()):
        Lat, Lon = float(r.lat), float(r.lon)
        lat.append(Lat);  lon.append(Lon)
        H = re.findall(getHashTags, r.tweet)
        TG = []
        if H:
            for tag in H:
                dTag = str(tag).strip().lower()
                if len(dTag)>2:
                    add = True
                    for fil in HTfilters:
                        if fil in dTag:
                            add=False; break
                    if add:
                        TG.append('#'+dTag)
            hashTags.append(TG)
        else:
            hashTags.append(TG)
        tweets.append(r.tweet)

    count = [1]*len(lat)
    df_loc = pd.DataFrame({'lat':lat, 'lon':lon, 'count':count, 'hashTags':hashTags, 'tweet':tweets})
    return df_loc

def haversine(lat1=0.0, lon1=0.0, lat2=0.0, lon2=0.0):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def tagsMap(df_loc):
    prov = pd.read_csv("data/provinsi-latlon.csv")
    ht_pos = {p:(0.0, 0.0, '') for p in prov.propinsi.to_list()}
    ht_tweets = {}
    for i, d in tqdm(df_loc.iterrows()):
        min_dis = ('', float('Inf'), 0.0, 0.0)
        for j, p in prov.iterrows():
            jrk = haversine(lat1=d.lat, lon1=d.lon, lat2=p.lat, lon2=p.lon)
            if jrk < min_dis[1]:
                min_dis = (p['propinsi'], jrk, p.lat, p.lon)
        ht_pos[min_dis[0]] = (min_dis[2], min_dis[3], ht_pos[min_dis[0]][2] + ' ' + ' '.join(d.hashTags))
        if '#' in d.tweet:
            try:
                ht_tweets[min_dis[0]].append(d.tweet)
            except:
                ht_tweets[min_dis[0]]= [d.tweet]

    for propinsi, dt in tqdm(ht_pos.items()):
        try:
            txt = dt[2]
            wc = WordCloud(max_font_size=75, min_font_size=16, max_words=3, background_color="rgba(0, 0, 255, 0)", color_func=lambda *args, **kwargs: (0,0,0), mode="RGBA").generate(txt)
            p = wc.to_file('data/clouds/{}.png'.format(propinsi))
        except:
            pass

    return ht_pos

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
        if f[-3:].lower() in ['txt', 'dic','py', 'ipynb']:
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print('error reading{0}'.format(f))
        else:
            print('Unsupported format {0}'.format(f))
    if file:
        Docs = Docs[0]
    return Docs, Files

def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        from spacy.lang.en import English as lemmatizer
        #lemmatizer = spacy.lang.en.English
        lemmatizer = lemmatizer()
        #lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_en.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        from spacy.lang.id import Indonesian
        #lemmatizer = spacy.lang.id.Indonesian
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer

def loadCorpus(file='', sep=':', dictionary = True):
    file = open(file, 'r', encoding="utf-8", errors='replace')
    F = file.readlines()
    file.close()
    if dictionary:
        fix = {}
        for f in F:
            k, v = f.split(sep)
            k, v = k.strip(), v.strip()
            fix[k] = v
    else:
        fix = set( (w.strip() for w in F) )
    return fix

slangFixId = loadCorpus(file = 'data/slang.dic', sep=':')
stopId, _ = LoadStopWords(lang='id')
stopId.add("rt")
def cleanTweet(data):
    cleanTweet = []
    for i, d in tqdm(data.iterrows()):
        cleanTweet.append(cleanText(d["tweet"], fix=slangFixId, lan='id', stops = stopId))
    return cleanTweet

def getTopic(df, Top_Words=30, resume_ = True, k=0):
    data_ta = df #df['clean'].values
    data = [t.split() for t in data_ta]
    
    if k==0:
        bigram_t = Phrases(data, min_count=2)
        trigram_t = Phrases(bigram_t[data])
        for idx, d in enumerate(data):
            for token in bigram_t[d]:
                if '_' in token:# Token is a bigram, add to document.
                    data[idx].append(token)
            for token in trigram_t[d]:
                if '_' in token:# Token is a bigram, add to document.
                    data[idx].append(token)
    
        dictionary_t = Dictionary(data)
        dictionary_t.filter_extremes(no_below=5, no_above=0.90)
        corpus_t = [dictionary_t.doc2bow(doc) for doc in data]
        start, step, limit = 2, 1, 5 # Ganti dengan berapa banyak Topic yang ingin di hitung/explore
        coh_t, kCV = [], 3 # hati-hati sangat lambat karena cross validation pada metode yang memang tidak efisien (LDA)
    
        for i in tqdm(range(kCV)):
            if resume_:
                try:
                    f = open('data/kCV_{}.pckl'.format(i), 'rb')
                    c = pickle.load(f); f.close()
                    coh_t.append(c)
                except:
                    model_list, c = compute_coherence_values(dictionary=dictionary_t, corpus=corpus_t, texts=data, start=start, limit=limit, step=step)
                    f = open('data/kCV_{}.pckl'.format(i), 'wb')
                    pickle.dump(c, f); f.close()
                    coh_t.append(c)
            else:
                model_list, c = compute_coherence_values(dictionary=dictionary_t, corpus=corpus_t, texts=data, start=start, limit=limit, step=step)
                f = open('data/kCV_{}.pckl'.format(i), 'wb')
                pickle.dump(c, f); f.close()
                coh_t.append(c)
    
        ct = np.mean(np.array(coh_t), axis=0).tolist()
        k = ct.index(max(ct))+start
    tf_w, tm_w, vec_w = getTopics(data_ta, n_topics=k, Top_Words=30)
    return tf_w, tm_w, vec_w, ct

ct = CRFTagger()  # Language Model
fTagger = 'data/all_indo_man_tag_corpus_model.crf.tagger'
ct.set_model_file(fTagger)

nlp_en = spacy.load("en_core_web_sm")
lemma_id = StemmerFactory().create_stemmer()

def connect(con="twitter", key=None, verbose=True):
    if con.lower().strip() == "twitter":
        Ck, Cs, At, As = key
        try:
            auth = tweepy.auth.OAuthHandler(Ck, Cs)
            auth.set_access_token(At, As)
            api = tweepy.API(auth, wait_on_rate_limit=True, timeout=180, retry_count=5, retry_delay=3)
            if verbose:
                usr_ = api.verify_credentials()
                print('Welcome "{}" you are now connected to twitter server'.format(usr_.name))
            return api
        except:
            print("Connection failed, please check your API keys or connection")

def crawlTwitter(api, qry, N = 30, lan='id', loc=None):
    T = []
    if loc:
        print('Crawling keyword "{}" from "{}"'.format(qry, loc))
        for tweet in tqdm(tweepy.Cursor(api.search_tweets, lang=lan, q=qry, count=100, tweet_mode='extended', geocode=loc).items(N)):
            T.append(tweet._json)
    else:
        print('Crawling keyword "{}"'.format(qry))
        for tweet in tqdm(tweepy.Cursor(api.search_tweets, q=qry, lang=lan, count=100, tweet_mode='extended').items(N)):
            T.append(tweet._json)
    print("Collected {} tweets".format(len(T)))
    return T

def getLatLon(gKey, location, lan='id'):
    gmaps = googlemaps.Client(key=gKey)
    try:
        res = gmaps.geocode(location, language=lan)[0]
    except Exception as err_:
        print(err_)
        return None, None, None
    if res:
        lat, lon = res['geometry']['location']['lat'], res['geometry']['location']['lng']
        addresses = res['address_components']
        alamat = [a['long_name'] for a in addresses]
        return lat, lon, alamat

def adaAngka(s):
    return any(i.isdigit() for i in s)

def fixTags(t):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    tagS = re.findall(getHashtags, t)
    for tag in tagS:
        if len(tag)>0:
            tg = tag[0].upper()+tag[1:]
            proper_words = []
            if adaAngka(tg):
                tag2 = re.split('(\d+)',tg)
                tag2 = [w for w in tag2 if len(w)>0]
                for w in tag2:
                    try:
                        _ = int(w) # error if w not a number
                        proper_words.append(w)
                    except:
                        w = w[0].upper()+w[1:]
                        proper_words = proper_words+re.findall(pisahtags, w)
            else:
                proper_words = re.findall(pisahtags, tg)
            proper_words = ' '.join(proper_words)
            t = t.replace('#'+tag, proper_words)
    return t

def loadCorpus(file='', sep=':', dictionary = True):
    file = open(file, 'r', encoding="utf-8", errors='replace')
    F = file.readlines()
    file.close()
    if dictionary:
        fix = {}
        for f in F:
            k, v = f.split(sep)
            k, v = k.strip(), v.strip()
            fix[k] = v
    else:
        fix = set( (w.strip() for w in F) )
    return fix

def crawlFiles(dPath,types=None): # dPath ='C:/Temp/', types = 'pdf'
    if types:
        return [dPath+f for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [dPath+f for f in os.listdir(dPath)]

def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        from spacy.lang.en import English as lemmatizer
        #lemmatizer = spacy.lang.en.English
        lemmatizer = lemmatizer()
        #lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_en.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        from spacy.lang.id import Indonesian
        #lemmatizer = spacy.lang.id.Indonesian
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer

def cleanText(T, fix={}, onlyChar=True, lemma=False, lan='id', stops = set(), symbols_remove = True, min_charLen = 2, max_charLen = 15, fixTag= True):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    if fixTag:
        t = fixTags(t) # fix abcDef
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        if symbols_remove:
            listKata = re.sub(r'[^.,_a-zA-Z0-9 -\.]',' ',K)

        listKata = TextBlob(listKata).words
        if fix:
            for j, token in enumerate(listKata):
                if str(token) in fix.keys():
                    listKata[j] = fix[str(token)]

        if onlyChar:
            listKata = [tok for tok in listKata if sum([1 for d in tok if d.isdigit()])==0]

        if stops:
            listKata = [tok for tok in listKata if str(tok) not in stops and len(str(tok))>=min_charLen]
        else:
            listKata = [tok for tok in listKata if len(str(tok))>=min_charLen]

        if lemma and lan.lower().strip()=='id':
            t[i] = lemma_id.stem(' '.join(listKata))
        elif lemma and lan.lower().strip()=='en':
            listKata = [str(tok.lemma_) for tok in nlp_en(' '.join(listKata))]
            t[i] = ' '.join(listKata)
        else:
            t[i] = ' '.join(listKata)

    return ' '.join(t) # Return kalimat lagi


def NLPfilter(t, filters):
    # filters = set(['NN', 'NNP', 'NNS', 'NNPS', 'JJ'])
    #tokens = TextBlob(t).words#nlp_id(t)
    tokens = [str(k) for k in TextBlob(t).words if len(k)>2]
    hasil = ct.tag_sents([tokens])
    return [k[0] for k in hasil[0] if k[1] in filters]

def compute_coherence_values(dictionary, corpus, texts, limit, coherence='c_v', start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def print_Topics(model, feature_names, Top_Topics, n_top_words):
    for topic_idx, topic in enumerate(model.components_[:Top_Topics]):
        print("Topic #%d:" %(topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def getTopics(Txt,n_topics=5, Top_Words=7):
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode', token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.95, min_df = 2)
    dtm_tf = tf_vectorizer.fit_transform(Txt)
    tf_terms = tf_vectorizer.get_feature_names()
    lda_tf = LDA(n_components=n_topics, learning_method='online', random_state=0).fit(dtm_tf)
    vsm_topics = lda_tf.transform(dtm_tf)
    doc_topic =  [a.argmax()+1 for a in vsm_topics] # topic of docs
    print('In total there are {0} major topics, distributed as follows'.format(len(set(doc_topic))))
    fig4 = plt.figure(); fig4.add_subplot(111)
    plt.hist(np.array(doc_topic), alpha=0.5); plt.show()
    print('Printing top {0} Topics, with top {1} Words:'.format(n_topics, Top_Words))
    print_Topics(lda_tf, tf_terms, n_topics, Top_Words)
    return lda_tf, dtm_tf, tf_vectorizer


def safeVectorizer(D, type_='tf', maxDf=0.95, minDf=2, ngram_=(1, 2)):
    vectorizer = CountVectorizer(binary = False, lowercase=True, max_df=maxDf, min_df=minDf)
    while True:
        X = vectorizer.fit_transform(D)
        if X[X.getnnz(1)>0].shape[0]==X.shape[0]:
            break
        else:
            newD = []
            nBaris, nKolom = X.shape
            for i,d in enumerate(D):
                if sum(X[i].data)!=0:
                    newD.append(d)
            D = newD
    return X, vectorizer.get_feature_names()

def drawGraph(G, Label, layOut='spring'):
    fig3 = plt.figure(); fig3.add_subplot(111)
    if layOut.lower()=='spring':
        pos = nx.spring_layout(G)
    elif layOut.lower()=='circular':
        pos=nx.circular_layout(G)
    elif layOut.lower()=='random':
        pos = nx.random_layout(G)
    elif layOut.lower()=='shells':
        shells = [G.core_nodes,sorted(G.major_building_routers, key=lambda n: nx.degree(G.topo, n)) + G.distribution_routers + G.server_nodes,G.hosts + G.minor_building_routers]
        pos = nx.shell_layout(G, shells)
    elif layOut.lower()=='spectral':
        pos=nx.spectral_layout(G)
    else:
        print('Graph Type is not available.')
        return
    nx.draw_networkx_nodes(G,pos, alpha=0.2,node_color='blue',node_size=600)
    if Label:
        nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edges(G,pos,width=4)
    plt.show()

def Graph(Tweets, Label = False, layOut='spring'): # Need the Tweets Before cleaning
    print("Please wait, building Graph .... ")
    G=nx.Graph()
    for tweet in tqdm(Tweets):
        if tweet['user']['screen_name'] not in G.nodes():
            G.add_node(tweet['user']['screen_name'])
        mentionS =  re.findall("@([a-zA-Z0-9]{1,15})", tweet['full_text'])
        for mention in mentionS:
            if "." not in mention: #skipping emails
                usr = mention.replace("@",'').strip()
                if usr not in G.nodes():
                    G.add_node(usr)
                G.add_edge(tweet['user']['screen_name'],usr)
    Nn, Ne = G.number_of_nodes(), G.number_of_edges()
    drawGraph(G, Label, layOut)
    print('Finished. There are %d nodes and %d edges in the Graph.' %(Nn,Ne))
    return G

def Centrality(G, N=10, method='katz', outliers=False, Label = True, layOut='shells'):

    if method.lower()=='katz':
        phi = 1.618033988749895 # largest eigenvalue of adj matrix
        ranking = nx.katz_centrality_numpy(G,1/phi)
    elif method.lower() == 'degree':
        ranking = nx.degree_centrality(G)
    elif method.lower() == 'eigen':
        ranking = nx.eigenvector_centrality_numpy(G)
    elif method.lower() =='closeness':
        ranking = nx.closeness_centrality(G)
    elif method.lower() =='betweeness':
        ranking = nx.betweenness_centrality(G)
    elif method.lower() =='harmonic':
        ranking = nx.harmonic_centrality(G)
    elif method.lower() =='percolation':
        ranking = nx.percolation_centrality(G)
    else:
        print('Error, Unsupported Method.'); return None

    important_nodes = sorted(ranking.items(), key=operator.itemgetter(1))[::-1]#[0:Nimportant]
    data = np.array([n[1] for n in important_nodes])
    dnodes = [n[0] for n in important_nodes][:N]
    if outliers:
        m = 1 # 1 standard Deviation CI
        data = data[:N]
        out = len(data[abs(data - np.mean(data)) > m * np.std(data)]) # outlier within m stDev interval
        if out<N:
            dnodes = [n for n in dnodes[:out]]

    print('Influencial Users: {0}'.format(str(dnodes)))
    print('Influencial Users Scores: {0}'.format(str(data[:len(dnodes)])))
    Gt = G.subgraph(dnodes)
    return Gt