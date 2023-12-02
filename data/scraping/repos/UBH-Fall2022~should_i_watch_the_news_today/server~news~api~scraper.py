
from time import sleep
from tkinter import ALL
import requests
from datetime import date
import cohere 
import pandas as pd
from sklearn.model_selection import train_test_split

co = cohere.Client("XBiccLrzkeP1yIlo8WgnjArtqSj2pzXcHfFny78E")

searchTerms = ""
search = input("Enter search terms: ").split(' ')

for t in search[:-1]:
    searchTerms += t+" OR "

searchTerms += search[-1]

# DATE = date.today()
# DATE = DATE.strftime("2022-%m-%d")
dates = ["2022-10-01", "2021-11-01", "2022-01-01", "2021-09-26", "2022-09-07"]


articleDescriptions = []
articleContents = []

for DATE in dates:
    ALL_URL = f'https://api.goperigon.com/v1/all?apiKey={settings.API_KEY}&from={DATE}&sourceGroup=top25finance&sortBy=date' #&q={searchTerms}'
    for i in range(1,11):
        curr_url = ALL_URL+f"&page={i}"
        sleep(5)
        response = requests.get(curr_url)
        print('Response obtained for '+DATE)
        articles = response.json()['articles']
        print('Appending to articles for '+DATE)
        for article in articles:
            articleContents.append(article['content'])
            articleDescriptions.append(article['description'])

print('Converting to df')
data = {'content': articleContents, 'description': articleDescriptions}
df = pd.DataFrame(data=data)
df.to_csv('data.csv')
