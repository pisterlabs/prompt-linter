from requests_oauthlib import OAuth1Session
import json
import openai
import pandas as pd
import os
from dotenv import load_dotenv
import pandas as pd
import json
import requests
import re
from collections import Counter
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

url_status = "https://api.twitter.com/1.1/statuses/update.json"
api_key = os.getenv('api_key')
api_key_secret = os.getenv('api_key_secret')
api_access_token = os.getenv('api_access_token')
api_access_token_secret = os.getenv('api_access_token_secret')
print("chla")


def get_tweet(command):
    print("tweet chal gya")
    message = [
        {"role": "system", "content": "you are an ai developed to  manage the twitter handle of Harshit Singh you have to make tweets about interesting science and technology availble in the world "},
        {"role": "system", "content": " you have to make a new tweet everyday and here are some thumb rules to follow:"
                                      " 1. The tweets should about science, technology ,machine learning , deep learning and other tech of the medival times" 
                                      " 2. The tweet should not exceed the limit of twitter tweets i.e around 280 characters"
                                      " 3. It should be unique"                   
         
         },
        {"role": "user", "content":command}
    ]


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0.8,
        messages=message
    )
    return response['choices'][0]['message']['content']


def get_image(description):
    
    # Generate an image using OpenAI's Image API.
    response = openai.Image.create(
    
    prompt= description,
    n=1,
    size="1024x1024"
    )
    return response['data'][0]['url']



url = "https://api.twitter.com/2/tweets"

twitter = OAuth1Session(api_key,
                        client_secret=api_key_secret,
                        resource_owner_key=api_access_token,
                        resource_owner_secret=api_access_token_secret)

headers = {
    "Content-Type": "application/json",
}
command= "Generate a new tweet "

tweet = get_tweet(command)
print(tweet)
image_url= get_image(tweet)
print("image generate hogyi")
# Download the generated image.
image_response = requests.get(image_url)
with open('generated_image.jpg', 'wb') as f:
    f.write(image_response.content)

# Upload media using chunked upload.
url_media_upload = "https://upload.twitter.com/1.1/media/upload.json?media_category=tweet_image"
files={'media': open('generated_image.jpg', 'rb')}
params={'media_category': 'tweet_image'}
response_upload = twitter.post(url_media_upload, files=files, params=params)
media_id = response_upload.json()['media_id_string']

# Post tweet with media.
url_status_update = "https://api.twitter.com/1.1/statuses/update.json"
data_status ={
   'status': tweet,
   'media_ids': media_id
}
response_status_update = twitter.post(url_status_update, data=data_status)

if response_status_update.status_code != 200:
    raise Exception(f"Request returned an error: {response_status_update.status_code}, {response_status_update.text}")

print("Tweet posted successfully!")