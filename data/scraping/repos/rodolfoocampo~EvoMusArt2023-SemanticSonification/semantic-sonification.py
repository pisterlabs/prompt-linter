# Flask Application that returns a json with data

# Importing flask and jsonify
from flask import Flask, jsonify, make_response, render_template, send_from_directory
import json

# stuff for text generation

import pandas as pd
import openai
import requests
from bs4 import BeautifulSoup
import datetime
import sys
import tweepy


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
import gspread

openai.api_key = [YOUR_API_KEY]

user_request_nature = 'Describe briefly how these data make you feel and why:'
user_request_culture = 'Interpret the data emotionally:'
user_request_governance = 'Extract the emotional quality of the data:'
user_request_infrastructure = 'Extract in three words the emotional quality of the data and explain why:'
user_request_commerce = 'Interpret the data emotionally:'
user_request_fashion = 'Interpret the data emotionally:'

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# add credentials to the account

# get the credentials-outside.json file from the static folder
creds = ServiceAccountCredentials.from_json_keyfile_name(LINK_TO_YOUR_CREDENTIALS, scope)
client = gspread.authorize(creds)


"""
Make an API call that queries this url: http://api.weatherapi.com/v1/current.json?key=[YOUR_API_KEY]&q=Canberra, Australia&aqi=no 
"""
def get_weather_data(place):
  """
  This function returns a dataframe with the weather data
  """
  # get the weather data
  url = 'http://api.weatherapi.com/v1/current.json?key=[YOUR_API_KEY]&q={}&units=m'.format(place)
  response = requests.get(url)
  data = response.json()
  print("got weather data")
  return data

"""
A sample response from the weather api is as follows:
{
    "location": {
        "name": "Canberra",
        "region": "Australian Capital Territory",
        "country": "Australia",
        "lat": -35.28,
        "lon": 149.22,
        "tz_id": "Australia/Sydney",
        "localtime_epoch": 1658200712,
        "localtime": "2022-07-19 13:18"
    },
    "current": {
        "last_updated_epoch": 1658200500,
        "last_updated": "2022-07-19 13:15",
        "temp_c": 11.0,
        "temp_f": 51.8,
        "is_day": 1,
        "condition": {
            "text": "Partly cloudy",
            "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
            "code": 1003
        },
        "wind_mph": 11.9,
        "wind_kph": 19.1,
        "wind_degree": 170,
        "wind_dir": "S",
        "pressure_mb": 1027.0,
        "pressure_in": 30.33,
        "precip_mm": 0.0,
        "precip_in": 0.0,
        "humidity": 47,
        "cloud": 50,
        "feelslike_c": 9.1,
        "feelslike_f": 48.3,
        "vis_km": 10.0,
        "vis_miles": 6.0,
        "uv": 3.0,
        "gust_mph": 11.4,
        "gust_kph": 18.4
    }
}
"""

# get the current condition text


def get_current_condition(data):
  """
  This function returns a string with the current condition
  """
  # get the current condition text
  current_condition = data['current']['condition']['text']
  return current_condition



"""
According to this list, create a function that receives the current condition as returns the valence
If Sunny or Clear then Positive
If Partly cloudy or Partly cloudy then Neutral
If Cloudy or Cloudy then Neutral
If Overcast or Overcast then Negative
If Mist or Mist then Negative
If Patchy rain possible or Patchy rain possible then Negative
If Patchy snow possible or Patchy snow possible then Negative
If Patchy sleet possible or Patchy sleet possible then Negative
If Patchy freezing drizzle possible or Patchy freezing drizzle possible then Negative
If Thundery outbreaks possible or Thundery outbreaks possible then Negative
If Blowing snow or Blowing snow then Negative
If Blizzard or Blizzard then Negative
If Fog or Fog then Neutral
If Freezing fog or Freezing fog then Negative
If Patchy light drizzle or Patchy light drizzle then Neutral
If Light drizzle or Light drizzle then Negative
If Freezing drizzle or Freezing drizzle then Negative
If Heavy freezing drizzle or Heavy freezing drizzle then Negative
If Patchy light rain or Patchy light rain then Negative
If Light rain or Light rain then Negative
If Moderate rain at times or Moderate rain at times then Negative
If Moderate rain or Moderate rain then Negative
If Heavy rain at times or Heavy rain at times then Negative
If Heavy rain or Heavy rain then Negative
If Light freezing rain or Light freezing rain then Negative
If Moderate or heavy freezing rain or Moderate or heavy freezing rain then Negative
If Light sleet or Light sleet then Negative
If Moderate or heavy sleet or Moderate or heavy sleet then Negative
If Patchy light snow or Patchy light snow then Negative
If Light snow or Light snow then Negative
If Patchy moderate snow or Patchy moderate snow then Negative
If Moderate snow or Moderate snow then Negative
If Patchy heavy snow or Patchy heavy snow then Negative
If Heavy snow or Heavy snow then Negative
If Ice pellets or Ice pellets then Negative
If Light rain shower or Light rain shower then Negative
If Moderate or heavy rain shower or Moderate or heavy rain shower then Negative
If Torrential rain shower or Torrential rain shower then Negative
If Light sleet showers or Light sleet showers then Negative
If Moderate or heavy sleet showers or Moderate or heavy sleet showers then Negative
If Light snow showers or Light snow showers then Negative
If Moderate or heavy snow showers or Moderate or heavy snow showers then Negative
If Light showers of ice pellets or Light showers of ice pellets then Negative
If Moderate or heavy showers of ice pellets or Moderate or heavy showers of ice pellets then Negative
If Patchy light rain with thunder or Patchy light rain with thunder then Negative
If Moderate or heavy rain with thunder or Moderate or heavy rain with thunder then Negative
If Patchy light snow with thunder or Patchy light snow with thunder then Negative
If Moderate or heavy snow with thunder or Moderate or heavy snow with thunder then Negative
"""


def get_valence(current_condition):
  """
  This function returns a string with the valence
  """
  # get the valence
  if current_condition == "Sunny" or current_condition == "Clear":
    valence = "Positive"
  elif current_condition == "Partly cloudy" or current_condition == "Partly cloudy":
    valence = "Neutral"
  elif current_condition == "Cloudy" or current_condition == "Cloudy":
    valence = "Neutral"
  elif current_condition == "Overcast" or current_condition == "Overcast":
    valence = "Negative"
  elif current_condition == "Mist" or current_condition == "Mist":
    valence = "Negative"
  else:
    valence = "Negative"
  return valence

"""
Evaluate content function
"""

def evaluateContent(content):
    response = openai.Completion.create(
      engine="content-filter-alpha",
      prompt = "<|endoftext|>"+content+"\n--\nLabel:",
      temperature=0,
      max_tokens=1,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      logprobs=10
    )
    response_text = response["choices"][0]["text"]

"""
Data2language function
"""


def data2language(data_list, explanation, request, temp):
  """
  This function receives a dataframe with the data we want to turn into a narrative and an accompanying explanation to provide context on content and structure
  Arguments
  raw_data: dataframe
  explanation: what the data is for, and it has in the columns
  """

  # convert the list of lists into a string, this will be used to build the prompt
  data_str = ''
  for item in data_list:
    data_str += str(item) + '\n'
  

  prompt= explanation + '\n' + 'Data:\n' + data_str + '\n' + request
  
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    temperature=temp,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  response_text = response['choices'][0]['text']

  label = evaluateContent(response_text)
  times = 0
  while label != '1' and label != '2' and times < 5:
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      temperature=temp,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    response_text = response['choices'][0]['text']

    label = evaluateContent(response_text)

    times += 1

  return response_text




# define a route '/api' and function api() that returns a json. The json has seven variables. The first one is called valence
# the others, in order are: nature, culture, infrastructure, governance, commerce and fashion. 




# I nbeed to get data from six different data sources, each corresponding to a layer in the Pace Layers diagram by Stewart Brand and Brian Eno. 
# The data sources are:
# 1. Nature - CO2 data
# 2. Culture - Wikipedia
# 3. Infrastructure - Budget from Australians Government
# 4. Governance - Data from democracy
# 5. Commerce - Data from the S&P 500
# 6. Fashion - Twitter data

##############
# CO2 DATA
###############
def get_nature(user_request):
  url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt'
  page = requests.get(url)
  page
  # Parse the data
  soup = BeautifulSoup(page.content, 'html.parser')

  # Get the text
  text = soup.get_text()

  # Split the text into lines
  lines = text.split('\n')

  # Get the data
  data = []
  for line in lines:
    if line:
      if line[0] != '#':
          data.append(line.split())


  data[len(data)-1][3]

  # create a new list of lists from data, but that only has the elements 0,1, and 3 of each list, make sure the range is correct
  data_new = []
  for i in range(len(data)):
      data_new.append([data[i][0],data[i][1],data[i][3]])

  data_new

  # Each line in data_new is a month, year, and CO2 level. Create a new list that only has the last 36 months of data. 


  data_new_last_36 = []
  for i in range(len(data_new)-36,len(data_new)):
      data_new_last_36.append(data_new[i])

  data_new_last_36

  data_nature = data_new_last_36

  explanation_nature = "This data contains the last 36 months of CO2 measurements in the atmosphere. The first column is the year, the second the month and the third is the CO2 level in parts per million."
  request_nature = user_request

  text_nature = data2language(data_nature, explanation_nature, request_nature, .7)

  print("Nature: " + text_nature)
  return text_nature

###########
#### WIKIPEDIA DATA - CULTURE
##########
def get_culture(user_request):
  today = datetime.datetime.now()
  date = today.strftime('%Y/%m/%d')

  # get the date for yesterday
  yesterday = today - datetime.timedelta(days=1)
  date = yesterday.strftime('%Y/%m/%d')
  url = 'https://api.wikimedia.org/feed/v1/wikipedia/en/featured/' + date

  response = requests.get(url)
  data = response.json()

  top10_wiki = []

  for i in range(10):
    top10_wiki.append(data['mostread']['articles'][i]['title'].replace('_', ' '))


  data_culture = top10_wiki


  explanation_culture = "This data contains the titles for the most read wikipedia articles today."
  request_culture = user_request

  print(explanation_culture)
  print(request_culture)
  text_culture = data2language(data_culture, explanation_culture, request_culture, .9)
  print("Culture: " + text_culture)
  return text_culture


#############
##### INFRASTRUCTURE
#############

# Using the world bank API, get all the indicator codes for the infrastructure topic
def get_infrastructure(user_request):
  url_infrastructure = 'https://api.worldbank.org/v2/topic/9/indicator?format=json'
  response = requests.get(url_infrastructure)
  indicators_infrastructure = response.json()

  # list all the indicator ids for the infrastructure topic
  indicators_infrastructure_ids = []
  for i in range(len(indicators_infrastructure[1])):
    indicators_infrastructure_ids.append(indicators_infrastructure[1][i]['id'])


  # list all the indicator names for the infrastructure topic. This is used to create the prompt for the text generation
  indicators_infrastructure_names = []
  for i in range(len(indicators_infrastructure[1])):
    indicators_infrastructure_names.append(indicators_infrastructure[1][i]['name'])


  # create a list that contains the names and the ids of the indicators for the infrastructure topic
  indicators_infrastructure_list = []
  for i in range(len(indicators_infrastructure[1])):
    indicators_infrastructure_list.append([indicators_infrastructure_names[i],indicators_infrastructure_ids[i]])


  # for each ID, call the WorldBank API and get the value for Australia. This will be used to create the prompt for the text generation. 



  indicators_infrastructure_values = []
  for i in range(10):
    url_infrastructure_value = 'https://api.worldbank.org/v2/country/au/indicator/' + indicators_infrastructure_list[i][1] + '?format=json'
    response = requests.get(url_infrastructure_value)
    # each element in the list corresponds to each year
    # iterate over the years
    for j in range(len(response.json()[1])):
      # obtain the first element where the value is not null
      if response.json()[1][j]['value'] != None:
        # append the value to the list, and the year to the list
        indicators_infrastructure_values.append([indicators_infrastructure_list[i][0], response.json()[1][j]['value'], response.json()[1][j]['date']])
        break

  data_infrastructure = indicators_infrastructure_values

  explanation_infrastructure = "This data contains the latest infrastructure indicators from the worldbank for Australia."
  request_infrastructure = user_request

  text_infrastructure = data2language(data_infrastructure, explanation_infrastructure, request_infrastructure, .9)

  print("Infrastructure: " + text_infrastructure)
  return text_infrastructure

#############
##### GOVERNANCE - PUBLIC SECTOR
#############


"""
In the realm of governance, the most interesting trend 
in current times—besides the worldwide proliferation of democracy and the 
rule of law——is the rise of what is coming to be called the "social sector."  
The public sector is government, the private sector is business, and the social sector is the nongovernmental, 
nonprofit do-good organizations.  Supported by philanthropy and the toil of volunteers, they range from church charities, 
local land trusts, and disease support groups to global players like the International Red Cross and World Wildlife Fund.  
What they have in common is that they serve the larger, slower good.
"""


# Using the world bank API, get all the indicator codes for the public sector topic
def get_governance(user_request):
  url_governance = 'https://api.worldbank.org/v2/topic/13/indicator?format=json'
  response = requests.get(url_governance)
  indicators_governance = response.json()

  # list all the indicator ids for the governance topic
  indicators_governance_ids = []
  for i in range(len(indicators_governance[1])):
    indicators_governance_ids.append(indicators_governance[1][i]['id'])


  # list all the indicator names for the governance topic. This is used to create the prompt for the text generation
  indicators_governance_names = []
  for i in range(len(indicators_governance[1])):
    indicators_governance_names.append(indicators_governance[1][i]['name'])


  # create a list that contains the names and the ids of the indicators for the governance topic
  indicators_governance_list = []
  for i in range(len(indicators_governance[1])):
    indicators_governance_list.append([indicators_governance_names[i],indicators_governance_ids[i]])


  # for each ID, call the WorldBank API and get the value for Australia. This will be used to create the prompt for the text generation. 

  indicators_governance_values = []
  for i in range(1,11):
    url_governance_value = 'https://api.worldbank.org/v2/country/au/indicator/' + indicators_governance_list[i][1] + '?format=json'
    response = requests.get(url_governance_value)
    # each element in the list corresponds to each year
    # iterate over the years
    for j in range(len(response.json()[1])):
      # obtain the first element where the value is not null
      if response.json()[1][j]['value'] != None:
        # append the value to the list, and the year to the list
        indicators_governance_values.append([indicators_governance_list[i][0], response.json()[1][j]['value'], response.json()[1][j]['date']])
        break

  data_governance = indicators_governance_values


  explanation_governance = "This data contains the latest governance indicators from the worldbank for Australia."
  request_governance = user_request


  text_governance = data2language(data_governance, explanation_governance, request_governance, .9)

  print("Governance: " + text_governance)
  return text_governance


#############
##### COMMERCE
#############


# Using the world bank API, get all the indicator codes for the public sector topic
def get_commerce(user_request):
  url_commerce = 'https://api.worldbank.org/v2/topic/3/indicator?format=json'
  response = requests.get(url_commerce)
  indicators_commerce = response.json()

  # list all the indicator ids for the commerce topic
  indicators_commerce_ids = []
  for i in range(len(indicators_commerce[1])):
    indicators_commerce_ids.append(indicators_commerce[1][i]['id'])


  # list all the indicator names for the commerce topic. This is used to create the prompt for the text generation
  indicators_commerce_names = []
  for i in range(len(indicators_commerce[1])):
    indicators_commerce_names.append(indicators_commerce[1][i]['name'])


  # create a list that contains the names and the ids of the indicators for the commerce topic
  indicators_commerce_list = []
  for i in range(len(indicators_commerce[1])):
    indicators_commerce_list.append([indicators_commerce_names[i],indicators_commerce_ids[i]])


  # for each ID, call the WorldBank API and get the value for Australia. This will be used to create the prompt for the text generation. 

  indicators_commerce_values = []
  for i in range(30,39):
    url_commerce_value = 'https://api.worldbank.org/v2/country/au/indicator/' + indicators_commerce_list[i][1] + '?format=json'
    response = requests.get(url_commerce_value)

    # each element in the list corresponds to each year
    # iterate over the years
    for j in range(len(response.json()[1])):
      # obtain the first element where the value is not null
      if response.json()[1][j]['value'] != None:
        # append the value to the list, and the year to the list
        indicators_commerce_values.append([indicators_commerce_list[i][0], response.json()[1][j]['value'], response.json()[1][j]['date']])
        break


  data_commerce = indicators_commerce_values


  explanation_commerce = "This data contains the latest commerce indicators from the worldbank for Australia."
  request_commerce = user_request


  text_commerce = data2language(data_commerce, explanation_commerce, request_commerce, .9)

  print("Commerce: " + text_commerce)
  return text_commerce

######
## FASHION
#####
#####


def get_fashion(user_request):
  # Twitter API keys
  consumer_key = [YOUR_API_KEY]
  consumer_secret = [YOUR_CONSUMER_SECRET]
  access_token = [YOUR_ACCESS_TOKEN]
  access_token_secret = [YOUR_TOKEN_SECRET]

  # Authenticate
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth)

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth)
  # Get the tweets
  tweets = api.user_timeline(screen_name="ANUcybernetics", 
                            # 200 is the maximum allowed count
                            count=100,
                            include_rts = False,
                            # Necessary to keep full_text 
                            # otherwise only the first 140 words are extracted
                            tweet_mode = 'extended'
                            )

  # get the latest tweet

  latest_tweet = tweets[0].full_text
  max_favs = 0
  for tweet in tweets:
    if tweet.favorite_count > max_favs:
      max_favs = tweet.favorite_count
      most_faved_tweet = tweet.full_text


  data_fashion = latest_tweet

  explanation_fashion = "This data contains the most faved tweet from the twitter account ANUcybernetics."
  request_fashion = user_request

  text_fashion = data2language(data_fashion, explanation_fashion, request_fashion, .9)

  print("Fashion: " + text_fashion)
  return text_fashion



####
# Transform each text data into an embedding
####


def get_embedding(text, engine="text-similarity-davinci-001"):
   return openai.Embedding.create(input = [text], engine=engine)['data'][0]['embedding']


# Upload the texts to Google Sheet

def upload_to_sheet(text_list):
  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

  for i in range(len(text_list)):
    # upload the text with index i to the sheet in the column i, in the second row
    worksheet.update_cell(2, i+1, text_list[i])

app = Flask(__name__)

# return the embeddings as a json through the API

@app.route('/', methods=['GET'])
def index():
    
    
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/api')
def api():
    weather_data = get_weather_data("Canberra, Australia")
    current_condition = get_current_condition(weather_data)
   
    valence_weather = get_valence(current_condition)


    try:
      text_nature = get_nature(user_request_nature)
    except:
      print("error getting nature data, returning generic response")
      text_nature = "CO2 levels are too high for the current climate. Please try again later."

    try:
      text_culture = get_culture(user_request_culture)
    except:
      print("error getting culture data, returning generic response")
      text_culture = "Beyoncé is still queen."

    try:
      text_governance = get_governance(user_request_governance)
    except:
      print("error getting governance data, returning generic response")
      text_governance = "The current Australian PM is Anthony Albanese. He has proposed a new climate change policy. Will he deliver?"

    try:
      text_infrastructure = get_infrastructure(user_request_infrastructure)
    except:
      print("error getting infrastructure data, returning generic response")
      text_infrastructure = "There are more than 810,000 km of public roads in Australia."

    try:
      text_commerce = get_commerce(user_request_commerce)
    except:
      print("error getting commerce data, returning generic response")
      text_commerce = "The S&P 500 is up and down, investors are left feeling like clowns"
    try:
      text_fashion = get_fashion(user_request_fashion)
    except:
      print("error getting fashion data, returning generic response")
      text_fashion = "The latest tweet from ANUcybernetics and its response shows increases interest in cybernetics."
    
    text_list = [text_nature, text_culture, text_governance, text_infrastructure, text_commerce, text_fashion]

    try:
      upload_to_sheet(text_list)
    except:
      print("error uploading text to sheet")
    
    try:
      embedding_nature = get_embedding(text_nature, engine="text-similarity-ada-001")
      embedding_culture = get_embedding(text_culture, engine="text-similarity-ada-001")
      embedding_infrastructure = get_embedding(text_infrastructure, engine="text-similarity-ada-001")
      embedding_governance = get_embedding(text_governance, engine="text-similarity-ada-001")
      embedding_commerce = get_embedding(text_commerce, engine="text-similarity-ada-001")
      embedding_fashion = get_embedding(text_fashion, engine="text-similarity-ada-001")
    except:
      print("error getting embeddings")
    

    valence = {'valence': valence_weather}
    nature = {'nature': embedding_nature}
    culture = {'culture': embedding_culture}
    infrastructure = {'infrastructure': embedding_infrastructure}
    governance = {'governance': embedding_governance} 
    commerce = {'commerce': embedding_commerce}
    fashion = {'fashion': embedding_fashion}


    return jsonify(valence, nature, culture, infrastructure, governance, commerce, fashion)

# Create individual endpoint for each layer, returning the embedding for each


@app.route('/api/nature')
def api_nature():
  try:
    text_nature = get_nature(user_request_nature)
  except:
    print("error getting nature data, returning generic response")
    text_nature = "CO2 levels are too high for the current climate. Please try again later."

  try:
    embedding_nature = get_embedding(text_nature, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  nature = {'nature': embedding_nature}

  # upload to sheet

  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

 
    # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 1, text_nature)
  return jsonify(nature)
  


@app.route('/api/culture')
def api_culture():
  try:
    text_culture = get_culture(user_request_culture)
  except:
    print("error getting culture data, returning generic response")
    text_culture = "Beyoncé is still queen."

  try:
    embedding_culture = get_embedding(text_culture, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  culture = {'culture': embedding_culture}

  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

 
  # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 2, text_culture)

  return jsonify(culture)


@app.route('/api/infrastructure')
def api_infrastructure():
  try:
    text_infrastructure = get_infrastructure(user_request_infrastructure)
  except:
    print("error getting infrastructure data, returning generic response")
    text_infrastructure = "There are more than 810,000 km of public roads in Australia."

  try:
    embedding_infrastructure = get_embedding(text_infrastructure, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  infrastructure = {'infrastructure': embedding_infrastructure}

  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

 
    # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 4, text_infrastructure)
  return jsonify(infrastructure)


@app.route('/api/governance')
def api_governance():
  try:
    text_governance = get_governance(user_request_governance)
  except:
    print("error getting governance data, returning generic response")
    text_governance = "The current Australian PM is Anthony Albanese. He has proposed a new climate change policy. Will he deliver?"

  try:
    embedding_governance = get_embedding(text_governance, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  governance = {'governance': embedding_governance}

  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

 
    # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 3, text_governance)
  return jsonify(governance)


@app.route('/api/commerce')
def api_commerce():
  try:
    text_commerce = get_commerce(user_request_commerce)
  except:
    print("error getting commerce data, returning generic response")
    text_commerce = "The S&P 500 is up and down, investors are left feeling like clowns"

  try:
    embedding_commerce = get_embedding(text_commerce, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  commerce = {'commerce': embedding_commerce}


  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

 
    # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 5, text_commerce)

  return jsonify(commerce)


@app.route('/api/fashion')
def api_fashion():
  try:
    text_fashion = get_fashion(user_request_fashion)
  except:
    print("error getting fashion data, returning generic response")
    text_fashion = "The latest tweet from ANUcybernetics and its response shows increases interest in cybernetics."

  try:
    embedding_fashion = get_embedding(text_fashion, engine="text-similarity-ada-001")
  except:
    print("error getting embeddings")
    
  fashion = {'fashion': embedding_fashion}

  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

  # upload the text with index i to the sheet in the column i, in the second row
  worksheet.update_cell(2, 6, text_fashion)
  return jsonify(fashion)

@app.route('/api/valence')
def api_valence():
  try:
    weather_data = get_weather_data("Canberra, Australia")
    current_condition = get_current_condition(weather_data)
   
    valence_weather = get_valence(current_condition)
  except:
    print("error getting valence data, returning generic response")
    valence_weather = "The weather is currently sunny."

  valence = {'valence': valence_weather}
  return jsonify(valence)


@app.route('/texts', methods=['GET'])
def text():  
  sheet = client.open("soundSheetFull")
  
  # get the worksheet called texts
  worksheet = sheet.worksheet("texts")

  # read the texts from row 2, for column 0 to 6 from the worksheet and put them in a list
  text_list = []
  for i in range(1,7):
    text_list.append(worksheet.cell(2, i).value)

  # make a json response where each text from the list is one object in the array

  json = {
    'nature': text_list[0],
    'culture': text_list[1],
    'governance': text_list[2],
    'infrastructure': text_list[3],
    'commerce': text_list[4],
    'fashion': text_list[5]
  }
  response = make_response(
                json, 200)

  response.headers["Content-Type"] = "application/json"
  # allow CORS
  response.headers["Access-Control-Allow-Origin"] = "*"

  return response
