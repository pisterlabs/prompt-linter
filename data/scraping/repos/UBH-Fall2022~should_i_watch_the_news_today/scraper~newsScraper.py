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
from_dates = ["2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01" ]
to_dates = ["2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31" ]
API_KEY = "135b093b-d521-4e9b-99c6-4c6f6af707b8"

articleDescriptions = []
articleContents = []

for fromDate, toDate in zip(from_dates, to_dates):
    ALL_URL = f'https://api.goperigon.com/v1/all?apiKey={API_KEY}&from={fromDate}&to={toDate}&sourceGroup=top25finance' 

    for i in range(1,11):
        curr_url = ALL_URL+f"&page={i}"
        response = requests.get(curr_url)
        print('Response obtained for '+fromDate+' page '+ str(i))
        sleep(5)
        articles = response.json()['articles']
        for article in articles:
            articleDescriptions.append(article['description'])

print('Converting to df')
data = {'description': articleDescriptions}
df = pd.DataFrame(data=data)
df.to_csv('data.csv')
