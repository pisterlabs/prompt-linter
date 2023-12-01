import pandas as pd
from datetime import datetime
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv, find_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import openai

# load secret from env
load_dotenv(find_dotenv())

openai.api_key = os.environ.get("OPENAI_API_KEY")
youtube_key = os.environ.get('YOUTUBE_KEY')
api_key = youtube_key

youtube = build('youtube', 'v3', developerKey=api_key)

# Replace 'your_keyword' with the keyword you want to search

max_results = 1000
serach_keyword = 'Desk setup 2023'

video_id = []
title = []
channel = []
release_date = []


request = youtube.search().list(
    part='snippet',
    maxResults=max_results,  
    q=serach_keyword,
    publishedAfter='2023-01-01T00:00:00Z', # get only result after 2023
    relevanceLanguage = 'en',
    type = 'video'
)
# for item in response['items']:
#     print('Video ID: ', item['id']['videoId'])
#     print('Title: ', item['snippet']['title'])
#     print('Channel: ', item['snippet']['channelTitle'])
#     print('---------------------------')

# storage results in a dataframe
# use for loop to get 50 reuslts each time
for _ in range(max_results // 50):
    response = request.execute()
    next_page_token = response.get('nextPageToken')

    request = youtube.search().list(
        part='snippet',
        maxResults=max_results,  
        q=serach_keyword,
        publishedAfter='2023-01-01T00:00:00Z', # get only result after 2023
        relevanceLanguage = 'en',
        type = 'video',
        pageToken = next_page_token
    )
    # append result in lists
    for item in response['items']:
        if item['id']['videoId'] not in video_id:
            video_id.append(item['id']['videoId'])
            title.append(item['snippet']['title'])
            channel.append(item['snippet']['channelTitle'])
            release_date.append(item['snippet']['publishedAt'])
    
# create dataframe
df = pd.DataFrame({'video_id': video_id, 'title':title, 'channel':channel, 'release_date':release_date})

# fotmat the release date
date_formatter = lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
df['release_date'] = df['release_date'].apply(date_formatter)

print(df.shape)
# get video transcript

def get_transcripts(video_ids:str) -> list:
    count = 0
    transcripts = []
    for video_id in video_ids:
        print(f'Getting transcript for video {count}')
        count += 1
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcripts.append(transcript)
        except:
            transcripts.append(None)

    return transcripts

# get transcript for each video_id in df
df['transcript'] = get_transcripts(df['video_id'])

# clean transcript column
# if transcript is None, drop the row
df = df.dropna(subset=['transcript'])
print(df.shape)

# remove depulicate video id
df['video_id'].drop_duplicates(inplace=True)

print(df.shape)

# get transcript text
df['transcript_text'] = df['transcript'].apply(lambda x: ' '.join([item['text'] for item in x]))
# drop transcript column
df = df.drop('transcript', axis=1)
# pass transcript text to openai api and get desk items from transcript

# save df to csv first
df.to_csv('./data/youtube-desk-setup-raw-data.csv', index=False)