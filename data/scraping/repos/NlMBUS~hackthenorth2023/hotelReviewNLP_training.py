import os
from sentence_transformers import SentenceTransformer
import cohere
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import pickle
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

#device = 'cuda' # nvidia-gpu
# device = 'mps' # apple-gpu
device = 'cpu' # no gpu

co = cohere.Client(os.getenv("COHERE_API_KEY")) # or None if you dont want to use Cohere

def encode(text):
  if co is not None:
    if len(text) > 95:
      embed = []
      # prod key is 10000 per minute, free is 100. Cohere offers $300 in credits using htn2023
      sleep_time = 60 / 100
      k = 0
      start = time.time()
      for i in tqdm(range(0, len(text), 95)):
        embed += co.embed(texts=text[i:i + 95]).embeddings
        k += 1
        if k == 100:
          end = time.time()
          dur = end - start
          time.sleep(60 - dur if 60 - dur > 0 else 0)
          start = time.time()
          k = 0
    else:
      embed = co.embed(
          texts=text,
          model='embed-english-v2.0'
      ).embeddings
    embed = np.array(embed)
  else:
    raise Exception("No API Key was found")
  
  return embed



df = pd.read_csv('../tripadvisor_hotel_reviews.csv')

encodings = encode(df["Review"].to_list())

df_train = df[:4000]
df_test = df[4000:5000]

# Prepare the training features and label
features = encodings[:4000]
label = df_train["Rating"]
inputs = encodings[4000:5000]



# c=1000 and gamma=0.0001 and kernel=rbf => 62.3%

# Tuning the Hyperparameters

'''
file = open("results.txt", "w")

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ["linear", "poly", "rbf", "sigmoid"]} 


for c_test in param_grid['C']:
  for gamma_test in param_grid['gamma']:
    for kernel_test in param_grid['kernel']:

      # Initialize the classifier
      # Scale the values -> Linear Classifier
      svm_classifier = make_pipeline(StandardScaler(), SVC(C=c_test, gamma=gamma_test, kernel=kernel_test))

      # Fit the support vector machine
      svm_classifier.fit(features, label)

      # Predict the labels
      df_test['Rating_pred'] = svm_classifier.predict(inputs)
      # Compute the score
      score = svm_classifier.score(inputs, df_test['Rating'])
      file.write(f"c={c_test} and gamma={gamma_test} and kernel={kernel_test} => {100*score}%\n")
      print(f"c={c_test} and gamma={gamma_test} and kernel={kernel_test} => {100*score}%\n")

'''
      
# Scale the values -> Linear Classifier
svm_classifier = make_pipeline(StandardScaler(), SVC(C=1000, gamma=0.0001, kernel="rbf"))

# Fit the support vector machine
svm_classifier.fit(features, label)

# Predict the labels
df_test['Rating_pred'] = svm_classifier.predict(inputs)
# Compute the score
score = svm_classifier.score(inputs, df_test['Rating'])

with open('model.pkl', 'wb') as f:
  pickle.dump(svm_classifier, f)

print(f"Accuracy: {100*score}%\n")

