from heyoo import WhatsApp
import requests
import re
import json
import urllib.request
import numpy as np
from geopy.geocoders import Nominatim
from geopy.point import Point
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)

lat, lon = 0,0
url = "https://maps.app.goo.gl/59Rc1YHQkvmnqTSD8"
session = requests.Session()  # so connections are recycled
resp = session.head(url, allow_redirects=True)
input_string = resp.url
pattern = re.compile(r'\b\d{1,2}\.\d{6},\d{1,2}\.\d{6}\b')
match = pattern.search(input_string)
coordstring = match.group(0).split(",")
coordstring = tuple(float(x) for x in coordstring)
lat, lon = coordstring[0], coordstring[1]
print(coordstring)



geolocator = Nominatim(user_agent="GPSregistros", timeout=200)

def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse(Point(lat, lon))
        return location.raw["display_name"]
    except:
        return None

location = reverse_geocoding(coordstring[0], coordstring[1])
print(location)
shortloc = location.split(" ")[1] + " "+ location.split(" ")[2]
city = location.split(" ")[1][:-1]
state = location.split(" ")[2]
print(city, state)
print(shortloc)



location = shortloc
article=""
articles = ""
apikey = ""
url = f"https://gnews.io/api/v4/search?q=disaster&lang=en&country=in&max=100&from=2023-11-01T01:52:00Z&to=2023-11-11T01:52:00Z&apikey={apikey}"

with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode("utf-8"))
    articles = data["articles"]

print(articles)
article = articles[0]["content"]
messenger = WhatsApp('',phone_number_id='180851955102577')
answer = "0"

#state = "Nepal"
for article in articles:
    print(article)
    articlecontent = article["content"]
    print(state)
    if state in articlecontent or city in articlecontent:
        prompt = f"Release a summary of the following article and the safety measures I should take considering I live nearby. The article is as follows. {article}"
        
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key="")
        messages = [SystemMessage(content=articlecontent), HumanMessage(content=prompt)]
        response = chat(messages)
        answer = response.content
        messenger.send_message(answer, 'number')
# For sending a Text messages
if answer=="0":
    messenger.send_message("No new info as of now", 'number')
