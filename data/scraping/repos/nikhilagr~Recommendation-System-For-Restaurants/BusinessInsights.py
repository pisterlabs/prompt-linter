import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
from string import punctuation
from nltk.corpus import stopwords
import nltk
import ssl
from time import time
from sklearn.ensemble import ExtraTreesRegressor

# from textblob import TextBlob
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from pprint import pprint
from sklearn.linear_model import Ridge
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# Enable logging for gensim - optional
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report
from datetime import datetime

class TopRecommendation:

    def __init__(self):
        self.reviews_filename = 'Finalized_Reviews.csv'
        self.users_filename = 'Finalized_users.csv'
        self.restaurants_filename = 'Finalized_Restaurants.csv'
        

    def prepareData(self):
        reviewData = pd.read_csv(self.reviews_filename)
        userData = pd.read_csv(self.users_filename)
        restaurantData = pd.read_csv(self.restaurants_filename)
        restaurantData = restaurantData.loc[restaurantData['business_id'].isin(reviewData['business_id'])]
        userData = userData.loc[userData['user_id'].isin(reviewData['user_id'])]
        return reviewData,userData,restaurantData


    def getSentimentScore(self,reviewData):
        sid = SentimentIntensityAnalyzer()
        pos = []
        neg = []
        for text in reviewData['text']:
            score = sid.polarity_scores(text)
            pos.append(score['pos'])
            neg.append(score['neg'])
        reviewData['PostiveScore'] = pos
        reviewData['NegativeScore'] = neg
        return reviewData


    def filterLen(self,docs, minlen):
        r""" filter out terms that are too short. 
        docs is a list of lists, each inner list is a document represented as a list of words
        minlen is the minimum length of the word to keep
        """
        return [ [t for t in d if len(t) >= minlen ] for d in docs ]

    def remove_stop_words(self,docs):
        en_stops = stopwords.words('english')
        en_stops.extend(['should','they','this','came','would','could'])
        new_docs = []
        for doc in docs:
            new_word = []  
            for word in doc:
                if word not in en_stops:
                    new_word.append(word)
            new_docs.append(new_word)
                
        return new_docs

    def filterInput(self,documents):
        new_docs = []
        for doc in documents:
            new_word = []
            for word in doc:
                new_word.append(word.lower())
                for char in word:
                    if(not char.isalpha()):
                        new_word.remove(word.lower())
                        break
            new_docs.append(new_word)
        
        return new_docs

    
    def remove_punctuation(self,docs):
        new_docs = []
        for doc in docs:
            new_words = []  
            for word in doc:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
            new_docs.append(new_words)
                
        return new_docs


    def preprocessText(self,reviewData):
        docs =  list(reviewData['text'])
        docs_tokens = [word_tokenize(doc) for doc in docs] 
        docs_filt = self.filterInput(docs_tokens)
        docs_wo_punctuation = self.remove_punctuation(docs_filt)
        preprocessed_docs = self.remove_stop_words(docs_wo_punctuation)
        return preprocessed_docs

    def createDictForLDA(self,preprocessed_docs):
        #Create Corpus
        texts = preprocessed_docs
        # Create Dictionary
        id2word = corpora.Dictionary(preprocessed_docs)
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        return id2word,corpus

    def getPredictedTopic(self,reviewData,lda_model,corpus):
        topic_pred = []
        for i in range(0, len(reviewData.text)):
            temp = lda_model[corpus[i]]
            result = sorted(temp,key=lambda x:(-x[1],x[0]))     
            topic_pred.append(result[0][0])
        reviewData['PredictedTopic'] = topic_pred
        return reviewData

    def createFeatureDF(self,reviewData):
        features = pd.DataFrame()
        features['PositiveScore'] = reviewData['PostiveScore']
        features['NegativeScore'] = reviewData['NegativeScore']
        return features

    def prepareFeatures(self,features):
        
        TP1 = []
        TP2 = []
        TP3 = []
        TP4 = []
        TP5 = []
        TP6 = []
        TP7 = []
        TP8 = []
        TP9 = []
        TP10 = []
        TN1 = []
        TN2 = []
        TN3 = []
        TN4 = []
        TN5 = []
        TN6 = []
        TN7 = []
        TN8 = []
        TN9 = []
        TN10 = []
        for j,row in reviewData.iterrows():
            ps = row['PostiveScore']
            ns = row['NegativeScore']
            temp = lda_model.get_topic_terms(row['PredictedTopic'])
            TP1.append(temp[0][1] * ps)
            TP2.append(temp[1][1] * ps)
            TP3.append(temp[2][1] * ps)
            TP4.append(temp[3][1] * ps)
            TP5.append(temp[4][1] * ps)
            TP6.append(temp[5][1] * ps)
            TP7.append(temp[6][1] * ps)
            TP8.append(temp[7][1] * ps)
            TP9.append(temp[8][1] * ps)
            TP10.append(temp[9][1] * ps)
            TN1.append(temp[0][1] * ns)
            TN2.append(temp[1][1] * ns)
            TN3.append(temp[2][1] * ns)
            TN4.append(temp[3][1] * ns)
            TN5.append(temp[4][1] * ns)
            TN6.append(temp[5][1] * ns)
            TN7.append(temp[6][1] * ns)
            TN8.append(temp[7][1] * ns)
            TN9.append(temp[8][1] * ns)
            TN10.append(temp[9][1] * ns)
        features['TP1'] = TP1
        features['TP2'] = TP2
        features['TP3'] = TP3
        features['TP4'] = TP4
        features['TP5'] = TP5
        features['TP6'] = TP6
        features['TP7'] = TP7
        features['TP8'] = TP8
        features['TP9'] = TP9
        features['TP10'] = TP10
        features['TN1'] = TN1
        features['TN2'] = TN2
        features['TN3'] = TN3
        features['TN4'] = TN4
        features['TN5'] = TN5
        features['TN6'] = TN6
        features['TN7'] = TN7
        features['TN8'] = TN8
        features['TN9'] = TN9
        features['TN10'] = TN10
        return features

    def predictRatings(self,features,reviewData):
        X = features
        y = reviewData['stars']
        # X_train , X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
        clf = Ridge(alpha=1.0)
        clf.fit(X, y) 
        pred_rating = clf.predict(X)
        reviewData['PredictedRating'] = pred_rating
        # reviewData.to_csv('PredictedRating')
        return reviewData

    def addMonthYear(self,reviewData):
        year = []
        month =[]
        for k,row in (reviewData).iterrows():
            dateobject = datetime.strptime(row['date'], '%Y-%m-%d')
            year.append(dateobject.year)
            month.append(dateobject.month)
        reviewData['year']= year
        reviewData['month']= month
        return reviewData

if __name__ == '__main__':
    start = time()
    tr = TopRecommendation()
    reviewData,userData,restaurantData = tr.prepareData()
    print(reviewData.head())
    reviewData = tr.getSentimentScore(reviewData)
    pre = tr.preprocessText(reviewData)
    id2word,corpus = tr.createDictForLDA(pre)
    lda_model = gensim.models.LdaMulticore(workers=3,corpus=corpus,id2word=id2word, num_topics=10, random_state=100,passes=5)
    reviewData = tr.getPredictedTopic(reviewData,lda_model,corpus)
    features = tr.createFeatureDF(reviewData)
    features = tr.prepareFeatures(features)
    reviewData = tr.predictRatings(features,reviewData)
    reviewData = tr.addMonthYear(reviewData)
    dataToPickle = [reviewData,restaurantData]
    pickle.dump(dataToPickle, open("modelTopRec.pkl","wb"))
    print("Model Dumped Successfully")
    end = time()
    print((end - start)/60)