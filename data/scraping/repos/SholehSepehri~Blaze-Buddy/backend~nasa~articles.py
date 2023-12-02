import cohere
from cohere.classify import Example
import pandas as pd
import pickle
import requests
import datetime
from tqdm import tqdm
from data import *
from datetime import date  
from datetime import timezone
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', None)

COHERE_API_KEY = "rKNbSEzFUz1Naxs2ZwQQpZOL3IsPAY4pKLIpLDnG"
NEWS_API_KEY = "603fbd47d85b463da271d5584b9701dc"
PREVIOUS_DATE = datetime.datetime.today() - datetime.timedelta(days=1)
PREVIOUS_DATE.date()

co = cohere.Client(COHERE_API_KEY)

from datetime import datetime

# HELPER FUNCTION
def get_location(location):
    pprompt = "Extract ONLY the location in the format city,province/state,country if available from the following text: " + location  + " .Else return None"
    response = co.generate(
                model='command-xlarge-nightly',              
                prompt=pprompt,
                max_tokens=300,
                temperature=0.3,
                k=0,
                p=0.75,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
    return response.generations[0].text

# EMBEDDING
df = pd.read_csv('training_data.csv', header=None, encoding='cp1252')

sentences_train, sentences_test, labels_train, labels_test = train_test_split(list(df[0]), list(df[1]), test_size=0.25, random_state=0)

# Embed the training set
embeddings_train = co.embed(texts=sentences_train, model="large", truncate="LEFT").embeddings

# Embed the testing set
embeddings_test = co.embed(texts=sentences_test, model="large", truncate="LEFT").embeddings

svm_classifier = make_pipeline(StandardScaler(), SVC(class_weight='balanced')) 

# fit the support vector machine
svm_classifier.fit(embeddings_train, labels_train)

score = svm_classifier.score(embeddings_test, labels_test)
print(f"Validation accuracy on the model is {100*score}%!")

# Hit news api on live news
url = "https://newsapi.org/v2/everything?q=burning+OR+fire+OR+flames+OR+wildfire&from=" + str(PREVIOUS_DATE) + "8&apiKey=" + NEWS_API_KEY
response = requests.get(url)
raw_data = response.text
data = json.loads(raw_data)

news_data = []
news_descriptions = []

for news in data["articles"]:
    news_descriptions.append(news["description"])
    d = datetime.fromisoformat(news["publishedAt"][:-1]).astimezone(timezone.utc)
    news_data.append([news["title"],d.strftime('%Y-%m-%d'),d.strftime('%I:%M %p')])    
    
y_test = co.embed(texts=news_descriptions, model="large", truncate="LEFT").embeddings
y_pred = svm_classifier.predict(y_test)

fire_data = []
for index in range(len(y_pred)):
    if y_pred[index] == "f":
        location = get_location(news_descriptions[index])
        print(location)
        if location != '\nNone':       
            merged_data = news_data[index] + [news_descriptions[index]] + [location]
            fire_data.append(merged_data)
print(fire_data)

pickle.dump(fire_data, open("news_fire_data.p", "wb"))