#!/usr/bin/env python3

# necessary libraries 
import os
import time
import math

import openai 
import pandas as pd 
import numpy as np 
# libraries to develop and evaluate a machine learning model 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import confusion_matrix 

openai.api_key = os.environ.get('OPENAI_API_KEY')

#get embedding function
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']

#load csv and ignore encoding errors and skip any bad line
df = pd.read_csv('dataset_source.csv',encoding_errors="ignore",on_bad_lines='skip')
print(); print(df.shape)
#drop cols with nulls
df = df.dropna(axis=1)
print();print(); print(df.head())

#print all
# Set display options to show all rows and columns
#pd.set_option('display.max_rows', None)    # Show all rows
#pd.set_option('display.max_columns', None) # Show all columns
#print(df)

#get a sample of the population
sample_size=50

#take the first {sample_size} datapoints
#df = df.iloc[:sample_size] 

#or

#take pct X% of spam, rest of ham
# Select 25 'ham' rows and 25 'spam' rows
pct_spam = 0.5
spam_rows = df[df['OUTPUT'] == 'spam'].head(math.ceil(sample_size*pct_spam))
ham_rows = df[df['OUTPUT'] == 'ham'].head(math.ceil(sample_size*(1-pct_spam)))
df = pd.concat([ham_rows, spam_rows])

filtered_df = df[df['OUTPUT'].isin(['ham', 'spam'])]
count_ham = (filtered_df['OUTPUT'] == 'ham').sum()
count_spam = (filtered_df['OUTPUT'] == 'spam').sum()
print(); print(df.head())
print();print();print(f"Ham:{count_ham} or {count_ham/len(filtered_df)*100:.2f}% - Spam:{count_spam} or {count_spam/len(filtered_df)*100:.2f}%")

#generate vectors for the dataset 
# #and convert the result to a string array and insert it into the dataset as a column
df["embedding"] = df.TEXT.apply(get_embedding).apply(np.array)
print(); print(df.head())

#relabel spam and ham
class_dict = {"spam":1,"ham":0}
df["class_embeddings"] = df.OUTPUT.map(class_dict)
#print(df.head())
print();print(); print(df)

#divide dataset into training and validation
x = np.array(df.embedding)
y = np.array(df.class_embeddings)
#20% of data for testing. by setting random_state to a fixed value, every time we run this it'll split the same way
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train random forest classifier with 100 decision trees
print(); print("Start to train the model.")
start_time = time.time()
clf = RandomForestClassifier(n_estimators=100)
#learn the relationship between features and class labels
clf.fit(x_train.tolist(),y_train)
end_time = time.time()
elapsed_time_seconds = end_time - start_time
minutes = int(elapsed_time_seconds // 60)
seconds = int(elapsed_time_seconds % 60)
milliseconds = int((elapsed_time_seconds - int(elapsed_time_seconds)) * 1000)
print(f"Time elapsed to train the model for {sample_size} mails: {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")

#predict the label
preds = clf.predict(x_test.tolist())

#generate a classification report involving f-1 score, recall, precision, accuracy
report = classification_report(y_test,preds)
print();print(report)
