import json
import requests
import bs4
BeautifulSoup = bs4.BeautifulSoup
import os
import openai
import boto3

openai.api_key = "" # enter your openai api key here
genius_key = "" # enter your genius api key here
song_url = "" # leave blank, used to generate song lyrics url

def get_lyrics(song_url):
    page = requests.get(song_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    lyrics = soup.find_all('div', {'class': 'Lyrics__Container-sc-1ynbvzw-10'})
    if len(lyrics) == 0:
        lyrics = soup.find_all('div', {'class': 'Lyrics__Container-sc-1ynbvzw-6'})
    raw_lyrics = ''
    for div in lyrics:
        for br in div.find_all("br"):
            br.replace_with("\n")
        raw_lyrics += div.text
    return raw_lyrics
    
def verse_count(lyrics):
    return lyrics.count("[Verse") + 1

def get_url(track_name, track_artist):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + genius_key}
    search_url = base_url + '/search'
    data = {'q': track_name + ' ' + track_artist}
    response = requests.get(search_url, params=data, headers=headers)
    json = response.json()
    remote_song_info = None
    for hit in json['response']['hits']:
        if track_artist.lower() in hit['result']['primary_artist']['name'].lower():
            remote_song_info = hit
            break
    return (remote_song_info['result']['url'], remote_song_info['result']['song_art_image_url'])

def write_lyrics(url):
    original_lyrics = get_lyrics(url)
    response = openai.Completion.create(
      engine="davinci",
      prompt="{}\n\n[Verse {}]".format(original_lyrics, verse_count(original_lyrics)),
      temperature=1.0,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.75,
      presence_penalty=0.5,
    )
    return response['choices'][0].text
    
def get_hash(lyrics):
    return abs(hash(lyrics))

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    lyrics_table = dynamodb.Table('lyrics')
    track_name = event['track_name']
    artist_name = event['artist_name']
    url = get_url(track_name, artist_name)
    song_url = url[0]
    image_url = url[1]
    try:
        lyrics = write_lyrics(song_url)
        verse_id = get_hash(lyrics)
        lyrics_table.put_item(
            Item={
                'verse_id': verse_id,
                'artist_name': artist_name,
                'track_name': track_name,
                'lyrics': lyrics
            }
        )
        
        return {
            'statusCode': 200,
            'body': {
                "lyrics": lyrics.replace("\n", "<br>"),
                "url": song_url,
                "image_url": image_url
            }
        }
    except Exception as e:
        print('Closing lambda function')
        return {
                'statusCode': 400,
                'body': json.dumps('Error writing the lyrics. {}'.format(e))
        }

    
    