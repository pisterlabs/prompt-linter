# Importing the libraries



import pandas as pd

import matplotlib.pyplot as plt

import os
# Loading the train data into pandas dataframe

# df_train = pd.read_csv('drive/My Drive/Jigsaw-CaseStudy/train.csv')

df_train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')



# We have 1.8 millions of data record in train dataset with 45 features given

print("Train dataframe shape:", df_train.shape)

df_train.head(5)
df_train.columns
# Loading the test data into pandas dataframe

# df_test = pd.read_csv('drive/My Drive/Jigsaw-CaseStudy/test.csv')

df_test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



# We have 10k of data record in test dataset 

print("Test dataframe shape:", df_test.shape)

df_test.head(5)
# For our case study we will focus on the comment_text data and rest of the columns we will ignore



df_train = df_train[['id','target','comment_text']]

df_train.head(5)
# Function to assign the binary class as y label



def assign_class(target):

    if target >= .5:

        return 1

    else: 

        return 0
# we will create binary class column which will be our Y label



df_train['class'] = df_train.apply(lambda x: assign_class(x['target']), axis= 1)

df_train.head(5)
# Total number of points with class 1 = 1,44,334 lakh



print("\nTotal number of points in both classes:")

df_train['class'].value_counts()
# EDA on target variable



fig = plt.figure(figsize=(10,10))

df_train.hist(column='target')

plt.xlabel("Target/Toxicity level")

plt.ylabel("No of comments")

plt.show()
# EDA on class variable



fig = plt.figure(figsize=(10,10))

df_train.hist(column='class')

plt.xlabel("Class")

plt.ylabel("No of comments")

plt.show()
import numpy as np



# Sampling the data such that both classes have equal number of datapoints in train dataset.



# https://stackoverflow.com/questions/56191448/sample-pandas-dataframe-based-on-values-in-column



df_train_sampled = df_train.groupby('class').apply(lambda x: x.sample(n=70000)).reset_index(drop = True)



print('\n Number of datapoints in each class :\n')

print(df_train_sampled['class'].value_counts())



print("\n The shape of train data is : ",df_train_sampled.shape)
df_train_sampled.head(5)
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
import string

import re

from nltk.stem.snowball import SnowballStemmer



def pre_process_text(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)

    

    ## Clean the text

    text = re.sub("[^a-zA-Z0-9\n]", " ", text) # removing special characters    

    text = re.sub("what's", "what is ", text) # decontracting the phrase    

    text = re.sub("\'ve", " have ", text) 

    text = re.sub("n't", " not ", text)

    text = re.sub("i'm", "i am ", text)

    text = re.sub("\'re", " are ", text)

    text = re.sub("\'d", " would ", text)

    text = re.sub("\'ll", " will ", text)

    text = re.sub("[.!#?]"," ", text)        

    text = re.sub("\s+"," ", text) # replace multiple spaces with single space

    

    ## Stemming

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    

    return text
import warnings

warnings.filterwarnings("ignore")



# Apply the preprocess text function to the train and test dataset



# Train dataset



df_train_sampled.comment_text.fillna(" ",inplace=True)



df_train_sampled['comment_text'] = df_train_sampled.comment_text.apply(lambda x: pre_process_text(x))

    

# Test dataset



df_test.comment_text.fillna(" ",inplace=True)



df_test['comment_text'] = df_test.comment_text.apply(lambda x: pre_process_text(x))
df_train_sampled.head(10)
df_test.head(10)
# References for Topic Modeling : 

# 1) https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# 2) https://nlpforhackers.io/topic-modeling/

# 3) https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28



data = df_train_sampled.comment_text.values.tolist()
import gensim

from gensim.utils import simple_preprocess

import warnings

warnings.filterwarnings("ignore")



def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
from gensim import corpora



# Create Dictionary

dictionary = corpora.Dictionary(data_words)



# Create Corpus

texts = data_words



# Term Document Frequency

corpus = [dictionary.doc2bow(text) for text in texts]



# View

print(corpus[:1])
# Human readable format of corpus (term-frequency)

[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]
import warnings

warnings.filterwarnings("ignore")



from gensim.models import LdaModel



# Build LDA model

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, update_every=1,

                                           chunksize=100, passes=10, alpha='auto', per_word_topics=True)
from gensim.models import CoherenceModel



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Print the Keyword in the 5 topics



print(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Visualize the topics on train



import pyLDAvis

import pyLDAvis.gensim 

import matplotlib.pyplot as plt


 

pyLDAvis.enable_notebook()



vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

vis
data = df_test.comment_text.values.tolist()

test_words = list(sent_to_words(data)) # corpus



print(test_words[:1])



# Term Document Frequency

test_corpus = [dictionary.doc2bow(text) for text in test_words]
vis = pyLDAvis.gensim.prepare(lda_model, test_corpus, dictionary)

vis
# For train vectors



train_vecs = []



for i in range(len(df_train_sampled)):

    top_train_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)

    topic_train_vec = [top_train_topics[i][1] for i in range(5)]

    train_vecs.append(topic_train_vec)
# Printing top five train vectors

train_vecs[:5]
# For test vectors



test_vecs = []



for i in range(len(df_test)):

    top_test_topics = lda_model.get_document_topics(test_corpus[i], minimum_probability=0.0)

    topic_test_vec = [top_test_topics[i][1] for i in range(5)]

    test_vecs.append(topic_test_vec)
# Printing top five test vectors

test_vecs[:5]
# Create the new df with 5 topics and then merge with the original train and test dataframe



df1 = pd.DataFrame(train_vecs,columns=['Topic-1','Topic-2','Topic-3','Topic-4','Topic-5'])

df_train_sampled = pd.concat([df_train_sampled, df1], axis=1, join='inner')



df2 = pd.DataFrame(test_vecs,columns=['Topic-1','Topic-2','Topic-3','Topic-4','Topic-5'])

df_test = pd.concat([df_test, df2], axis=1, join='inner')
df_train_sampled.head(5)
df_test.head(5)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tqdm import tqdm

import nltk

nltk.download('vader_lexicon')



sid = SentimentIntensityAnalyzer()



# ******************************************* Sentiment analysis for train data

text_NegScore = []

text_NeuScore = []

text_PosScore = []

text_compoundScore = []



for stmt in tqdm(df_train_sampled.comment_text.values):

    

    ss = sid.polarity_scores(str(stmt))

    text_NegScore.append(ss["neg"])

    text_NeuScore.append(ss["neu"])

    text_PosScore.append(ss["pos"])

    text_compoundScore.append(ss["compound"])



# Assigning the score to the train text

df_train_sampled['text_NegScore'] = text_NegScore

df_train_sampled['text_NeuScore'] = text_NeuScore

df_train_sampled['text_PosScore'] = text_PosScore

df_train_sampled['text_compoundScore'] = text_compoundScore



# ******************************************** Sentiment analysis for test data

text_NegScore = []

text_NeuScore = []

text_PosScore = []

text_compoundScore = []

    

for stmt in tqdm(df_test.comment_text.values):

    

    ss = sid.polarity_scores(str(stmt))

    text_NegScore.append(ss["neg"])

    text_NeuScore.append(ss["neu"])

    text_PosScore.append(ss["pos"])

    text_compoundScore.append(ss["compound"])



# Assigning the score to the test text data

df_test['text_NegScore'] = text_NegScore

df_test['text_NeuScore'] = text_NeuScore

df_test['text_PosScore'] = text_PosScore

df_test['text_compoundScore'] = text_compoundScore
df_train_sampled.head(5)
df_test.head(5)
# Count of the length if comment for train and test dataset



WrdCnt_trn = []; # the avg-w2v for each sentence/review is stored in this list

WrdCnt_test = [];



df_train_sampled.comment_text.fillna(" ",inplace=True)

df_test.comment_text.fillna(" ",inplace=True)



# For train data

for sentence in tqdm(df_train_sampled['comment_text'].values): # for each review/sentence

    cnt_words = 0; # num of words with a valid vector in the sentence/review

    cnt_words = sentence.split() # for each word in a review/sentence

    WrdCnt_trn.append(len(cnt_words))

    

# For test data

for sentence in tqdm(df_test['comment_text'].values): # for each review/sentence

    cnt_words = 0; # num of words with a valid vector in the sentence/review

    cnt_words = sentence.split() # for each word in a review/sentence

    WrdCnt_test.append(len(cnt_words))



print("Length of Comment text in train dataset :", len(WrdCnt_trn))

print("Length of Comment text in test dataset :", len(WrdCnt_test))
df_train_sampled['Comment_Len'] = WrdCnt_trn

df_test['Comment_Len'] = WrdCnt_test
df_train_sampled.head(2)
df_test.head(2)
from sklearn.model_selection import train_test_split



# Taking class as y label in differnt dataset and then dropping the target and class from train dataset.

Y_train = df_train_sampled['class']

df_train_sampled.drop(columns=['target','class'], axis=1, inplace=True)



# split the train data into train and cross validation by maintaining same distribution of output varaible 'Y_train' [stratify=Y_train]

# Train : cross validate ratio = 70 : 30



train_df, cv_df, y_train, y_cv = train_test_split(df_train_sampled, Y_train, stratify=Y_train, test_size=0.30)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])

print('Number of data points of Y label in train data:', y_train.shape[0])

print('Number of data points of Y label in cross validation data:', y_cv.shape[0])
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vect = TfidfVectorizer(ngram_range=(1,6),max_features=7000)



tfidf_train = tfidf_vect.fit_transform(train_df['comment_text'].values.astype('U'))

tfidf_cv = tfidf_vect.transform(cv_df['comment_text'].values.astype('U'))

tfidf_test = tfidf_vect.transform(df_test['comment_text'].values.astype('U'))



print('shape of train TFIDF vector : ', tfidf_train.shape)

print('shape of cross validate TFIDF vector : ', tfidf_cv.shape)

print('shape of test TFIDF vector : ', tfidf_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_charvect = TfidfVectorizer(ngram_range=(1,6), analyzer='char', max_features=7000)



tfidf_chartrain = tfidf_charvect.fit_transform(train_df['comment_text'].values.astype('U'))

tfidf_charcv = tfidf_charvect.transform(cv_df['comment_text'].values.astype('U'))

tfidf_chartest = tfidf_charvect.transform(df_test['comment_text'].values.astype('U'))



print('shape of train TFIDF vector : ', tfidf_chartrain.shape)

print('shape of cross validate TFIDF vector : ', tfidf_charcv.shape)

print('shape of test TFIDF vector : ', tfidf_chartest.shape)
# Vectorizing neg score 



from sklearn.preprocessing import MinMaxScaler



Neg_scaler = MinMaxScaler()



Neg_scaler.fit(train_df['text_NegScore'].values.reshape(-1,1))



neg_train = Neg_scaler.transform(train_df['text_NegScore'].values.reshape(-1,1))

neg_cv = Neg_scaler.transform(cv_df['text_NegScore'].values.reshape(-1,1))

neg_test = Neg_scaler.transform(df_test['text_NegScore'].values.reshape(-1,1))



print('Shape of neg_train :', neg_train.shape)

print('Shape of neg_cv :', neg_cv.shape)

print('Shape of neg_test :', neg_test.shape)
# Vectorizing pos score 



from sklearn.preprocessing import MinMaxScaler



Pos_scaler = MinMaxScaler()



Pos_scaler.fit(train_df['text_PosScore'].values.reshape(-1,1))



pos_train = Pos_scaler.transform(train_df['text_PosScore'].values.reshape(-1,1))

pos_cv = Pos_scaler.transform(cv_df['text_PosScore'].values.reshape(-1,1))

pos_test = Pos_scaler.transform(df_test['text_PosScore'].values.reshape(-1,1))



print('Shape of pos_train :', pos_train.shape)

print('Shape of pos_cv :', pos_cv.shape)

print('Shape of pos_test :', pos_test.shape)
# Vectorizing neu score 



from sklearn.preprocessing import MinMaxScaler



Neu_scaler = MinMaxScaler()



Neu_scaler.fit(train_df['text_NeuScore'].values.reshape(-1,1))



neu_train = Neu_scaler.transform(train_df['text_NeuScore'].values.reshape(-1,1))

neu_cv = Neu_scaler.transform(cv_df['text_NeuScore'].values.reshape(-1,1))

neu_test = Neu_scaler.transform(df_test['text_NeuScore'].values.reshape(-1,1))



print('Shape of neu_train :', neu_train.shape)

print('Shape of neu_cv :', neu_cv.shape)

print('Shape of neu_test :', neu_test.shape)
# Vectorizing compound score 



from sklearn.preprocessing import MinMaxScaler



Comp_scaler = MinMaxScaler()



Comp_scaler.fit(train_df['text_compoundScore'].values.reshape(-1,1))



comp_train = Comp_scaler.transform(train_df['text_compoundScore'].values.reshape(-1,1))

comp_cv = Comp_scaler.transform(cv_df['text_compoundScore'].values.reshape(-1,1))

comp_test = Comp_scaler.transform(df_test['text_compoundScore'].values.reshape(-1,1))



print('Shape of comp_train :', comp_train.shape)

print('Shape of comp_cv :', comp_cv.shape)

print('Shape of comp_test :', comp_test.shape)
# Vectorizing compound score 



from sklearn.preprocessing import StandardScaler



Len_scaler = StandardScaler()



Len_scaler.fit(train_df['Comment_Len'].values.reshape(-1,1))

Len_scaler.fit(cv_df['Comment_Len'].values.reshape(-1,1))

Len_scaler.fit(df_test['Comment_Len'].values.reshape(-1,1))



len_train = Len_scaler.transform(train_df['Comment_Len'].values.reshape(-1,1))

len_cv = Len_scaler.transform(cv_df['Comment_Len'].values.reshape(-1,1))

len_test = Len_scaler.transform(df_test['Comment_Len'].values.reshape(-1,1))



print('Shape of comp_train :', len_train.shape)

print('Shape of comp_cv :', len_cv.shape)

print('Shape of comp_test :', len_test.shape)
# Vectorizing topic-1



from sklearn.preprocessing import StandardScaler



topic_scaler = StandardScaler()



topic_scaler.fit(train_df['Topic-1'].values.reshape(-1,1))

topic_scaler.fit(cv_df['Topic-1'].values.reshape(-1,1))

topic_scaler.fit(df_test['Topic-1'].values.reshape(-1,1))



topic1_train = topic_scaler.transform(train_df['Topic-1'].values.reshape(-1,1))

topic1_cv = topic_scaler.transform(cv_df['Topic-1'].values.reshape(-1,1))

topic1_test = topic_scaler.transform(df_test['Topic-1'].values.reshape(-1,1))



print('Shape of topic1_train :', topic1_train.shape)

print('Shape of topic1_cv :', topic1_cv.shape)

print('Shape of topic1_test :', topic1_test.shape)
# Vectorizing topic-2



topic_scaler.fit(train_df['Topic-2'].values.reshape(-1,1))

topic_scaler.fit(cv_df['Topic-2'].values.reshape(-1,1))

topic_scaler.fit(df_test['Topic-2'].values.reshape(-1,1))



topic2_train = topic_scaler.transform(train_df['Topic-2'].values.reshape(-1,1))

topic2_cv = topic_scaler.transform(cv_df['Topic-2'].values.reshape(-1,1))

topic2_test = topic_scaler.transform(df_test['Topic-2'].values.reshape(-1,1))



print('Shape of topic2_train :', topic2_train.shape)

print('Shape of topic2_cv :', topic2_cv.shape)

print('Shape of topic2_test :', topic2_test.shape)
# Vectorizing topic-3



topic_scaler.fit(train_df['Topic-3'].values.reshape(-1,1))

topic_scaler.fit(cv_df['Topic-3'].values.reshape(-1,1))

topic_scaler.fit(df_test['Topic-3'].values.reshape(-1,1))



topic3_train = topic_scaler.transform(train_df['Topic-3'].values.reshape(-1,1))

topic3_cv = topic_scaler.transform(cv_df['Topic-3'].values.reshape(-1,1))

topic3_test = topic_scaler.transform(df_test['Topic-3'].values.reshape(-1,1))



print('Shape of topic3_train :', topic3_train.shape)

print('Shape of topic3_cv :', topic3_cv.shape)

print('Shape of topic3_test :', topic3_test.shape)
# Vectorizing topic-4



topic_scaler.fit(train_df['Topic-4'].values.reshape(-1,1))

topic_scaler.fit(cv_df['Topic-4'].values.reshape(-1,1))

topic_scaler.fit(df_test['Topic-4'].values.reshape(-1,1))



topic4_train = topic_scaler.transform(train_df['Topic-4'].values.reshape(-1,1))

topic4_cv = topic_scaler.transform(cv_df['Topic-4'].values.reshape(-1,1))

topic4_test = topic_scaler.transform(df_test['Topic-4'].values.reshape(-1,1))



print('Shape of topic4_train :', topic4_train.shape)

print('Shape of topic4_cv :', topic4_cv.shape)

print('Shape of topic4_test :', topic4_test.shape)
# Vectorizing topic-5



topic_scaler.fit(train_df['Topic-5'].values.reshape(-1,1))

topic_scaler.fit(cv_df['Topic-5'].values.reshape(-1,1))

topic_scaler.fit(df_test['Topic-5'].values.reshape(-1,1))



topic5_train = topic_scaler.transform(train_df['Topic-5'].values.reshape(-1,1))

topic5_cv = topic_scaler.transform(cv_df['Topic-5'].values.reshape(-1,1))

topic5_test = topic_scaler.transform(df_test['Topic-5'].values.reshape(-1,1))



print('Shape of topic5_train :', topic5_train.shape)

print('Shape of topic5_cv :', topic5_cv.shape)

print('Shape of topic5_test :', topic5_test.shape)
from scipy.sparse import hstack



X_train = hstack((tfidf_train,tfidf_chartrain,neg_train,neu_train,pos_train,comp_train,len_train,topic1_train,topic2_train,topic3_train,topic4_train,topic5_train))

X_cv = hstack((tfidf_cv,tfidf_charcv,neg_cv,neu_cv,pos_cv,comp_cv,len_cv,topic1_cv,topic2_cv,topic3_cv,topic4_cv,topic5_cv))

X_test = hstack((tfidf_test,tfidf_chartest,neg_test,neu_test,pos_test,comp_test,len_test,topic1_test,topic2_test,topic3_test,topic4_test,topic5_test))



print("Shape of X train dataset : ", X_train.shape)

print("Shape of CV dataset : ", X_cv.shape)

print("Shape of X test dataset : ", X_test.shape)
from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.metrics import roc_auc_score, auc



import warnings

warnings.filterwarnings("ignore")



train_auc = []

cv_auc = []



alpha = [10 ** x for x in range(-7, -1)]

cv_log_error_array = []

for i in alpha:

    

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)    



    y_predictedtrain = clf.predict(X_train)

    y_predictedCV = clf.predict(X_cv)



    train_auc.append(roc_auc_score(y_train, y_predictedtrain))

    cv_auc.append(roc_auc_score(y_cv, y_predictedCV))

    

fig, ax = plt.subplots()

ax.plot(alpha, train_auc, label="train AUC")

ax.plot(alpha, cv_auc, label="CV AUC")



ax.scatter(alpha, train_auc, label="train AUC pts")

ax.scatter(alpha, cv_auc, label="CV AUC pts")



plt.legend()

plt.xlabel("Alpha (Hyper parameter)")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



def plot_confusion_matrixas_HeatMap(confusion_matrix):

    col = ["Predicted project rejected","Prediction project approved"]

    ind = ["Actual project rejected","Actual project approved"]

    df_cm = pd.DataFrame(confusion_matrix, index=ind, columns=col)

    fig = plt.figure(figsize=(4,4))

    plt.close()

    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

    plt.ylabel('Predicted label')

    plt.xlabel('True label')

    return fig
from sklearn.metrics import confusion_matrix

# Confusion Matrix for cross validate data



# From Hyperparameter tuning we take that point where train and validate AUC points have high score

clf = SGDClassifier(class_weight='balanced', alpha=10**-6, penalty='l2', loss='log', random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)



y_predictedCV = clf.predict(X_cv)

cm = confusion_matrix(y_cv, y_predictedCV)

plot_confusion_matrixas_HeatMap(cm)
from sklearn.metrics import roc_curve, auc



y_predictedtrain = clf.predict(X_train)

y_predictedCV = clf.predict(X_cv)



train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_predictedtrain)

test_fpr, test_tpr, te_thresholds = roc_curve(y_cv, y_predictedCV)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="CV AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("ROC Curve for Train and CV data")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.metrics import roc_auc_score, auc



import warnings

warnings.filterwarnings("ignore")



train_auc = []

cv_auc = []



alpha = [10 ** x for x in range(-7, -1)]

cv_log_error_array = []

for i in alpha:

    

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)    



    y_predictedtrain = clf.predict(X_train)

    y_predictedCV = clf.predict(X_cv)



    train_auc.append(roc_auc_score(y_train, y_predictedtrain))

    cv_auc.append(roc_auc_score(y_cv, y_predictedCV))

    

fig, ax = plt.subplots()

ax.plot(alpha, train_auc, label="train AUC")

ax.plot(alpha, cv_auc, label="CV AUC")



ax.scatter(alpha, train_auc, label="train AUC pts")

ax.scatter(alpha, cv_auc, label="CV AUC pts")



plt.legend()

plt.xlabel("Alpha (Hyper parameter)")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
from sklearn.metrics import confusion_matrix

# Confusion Matrix for cross validate data



# From Hyperparameter tuning we take that point where train and validate AUC points have high score

clf = SGDClassifier(class_weight='balanced', alpha=10**-5, penalty='l2', loss='hinge', random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)



y_predictedCV = clf.predict(X_cv)

cm = confusion_matrix(y_cv, y_predictedCV)

plot_confusion_matrixas_HeatMap(cm)
from sklearn.metrics import roc_curve, auc



y_predictedtrain = clf.predict(X_train)

y_predictedCV = clf.predict(X_cv)



train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_predictedtrain)

test_fpr, test_tpr, te_thresholds = roc_curve(y_cv, y_predictedCV)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="CV AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("ROC Curve for Train and CV data")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
# http://zetcode.com/python/prettytable/



from prettytable import PrettyTable

    

x = PrettyTable()



x.field_names = ["Model","alpha","Train AUC", "Validate AUC"]



x.align["Vectorizer"] = "l"

x.align["Hyper parameter"] = "r"

x.align["Train AUC"] = "r"

x.align["Validate AUC"] = "r"



x.add_row(["logistic regresion", "alpha = 0.000001", "0.89", "0.85"])

x.add_row(["linear SVM", "alpha = 0.000001", "0.88", "0.84"])



print(x)
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix



param_grid = {

        'silent': [False],

        'max_depth': [10],

        'learning_rate': [0.1],

        'n_estimators': [300]}



clf = xgb.XGBClassifier()



rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100, n_jobs=-1, verbose=2, cv=5, refit=True, random_state=42)

rs_clf.fit(X_train,y_train)



y_predictedCV = rs_clf.predict(X_cv)

cm = confusion_matrix(y_cv, y_predictedCV)

plot_confusion_matrixas_HeatMap(cm)
from sklearn.metrics import roc_curve, auc



y_predictedtrain = rs_clf.predict(X_train)

y_predictedCV = rs_clf.predict(X_cv)



train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_predictedtrain)

test_fpr, test_tpr, te_thresholds = roc_curve(y_cv, y_predictedCV)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="CV AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("ROC Curve for Train and CV data")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
# As we can see that train AUC for XGBoost classifier is bestest among the logistic regression and linear SVM

# we will be predicting the test dataset using XGBoost



y_predicted_test = rs_clf.predict(X_test)

y_predict_proba_test = rs_clf.predict_proba(X_test)



print(y_predicted_test[:20])
df_test['comment_text'][11] 
# Scores for traning and cross validate data



print(rs_clf.score(X=X_train,y=y_train))

print(rs_clf.score(X=X_cv,y=y_cv))
print(y_predict_proba_test)
# Merging the predicted Y class and probability to the test dataframe



df_test['prediction'] = list(y_predicted_test)

df_test['Y_predictClass_probability'] = list(y_predict_proba_test)
df_test.head(20)
sample_submission_df = pd.DataFrame(columns=['id','prediction'])

sample_submission_df['id'] = df_test['id']

sample_submission_df['prediction'] = df_test['prediction']
sample_submission_df.to_csv('submission.csv',index=False)