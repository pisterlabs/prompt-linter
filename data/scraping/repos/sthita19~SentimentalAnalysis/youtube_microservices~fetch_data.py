from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
from langchain.document_loaders import YoutubeLoader
import streamlit as st
import tqdm
from tqdm import tqdm

def get_channel_id(youtube, channel_name):
    channel_id = ''
    try:
        # Fetch channel data based on the channel name
        search_response = youtube.search().list(
            part='id',
            q=channel_name,
            type='channel',
            maxResults=1,
        ).execute()

        # Get the channel ID from the search results
        if 'items' in search_response:
            channel_id = search_response['items'][0]['id']['channelId']
        else:
            print(f"No channel found with the name '{channel_name}'")

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')

    return channel_id

def get_video_urls(youtube, channel_id, start_date, end_date):
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    video_urls = []

    # Fetch video_id using YouTube Data API and building the url
    try:
        next_page_token = ''
        while True:
            playlist_response = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=25,
                publishedAfter=start_date_str + 'T00:00:00Z',
                publishedBefore=end_date_str + 'T23:59:59Z',
                pageToken=next_page_token
            ).execute()

            for item in playlist_response['items']:
                if item['id']['kind'] == 'youtube#video':
                    
                    video_id = item['id']['videoId']
                    video_title = item['snippet']['title']
                    st.text(f'Title: {video_title}')
                    video_url = f'https://www.youtube.com/watch?v={video_id}'
                    video_urls.append(video_url)

            if 'nextPageToken' in playlist_response:
                next_page_token = playlist_response['nextPageToken']
            else:
                break
            

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')

    return video_urls

# Getting transcripts from video urls
def get_transcripts(video_urls):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    transcripts = []
    i = 0
    with tqdm(total=len(video_urls), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for video_url in video_urls:
            loader = YoutubeLoader.from_youtube_url(
                video_url, add_video_info=True, language=['en', 'en-IN', 'hi'], translation='en'
            )
            transcript = loader.load()
            if transcript:
                transcript_content = transcript[0].page_content
                transcripts.append(transcript_content)
            
            # Update the progress bar
            progress = (i + 1) / len(video_urls)
            progress_bar.progress(progress)

            progress_percent = int(progress * 100)
            progress_text.text(f"Progress: {progress_percent}%")
            pbar.update(1)
            i += 1


    return transcripts

# Driver function
def execute(youtube, channel_name, start_date, end_date):
    channel_id = get_channel_id(youtube=youtube, channel_name=channel_name)
    video_urls = get_video_urls(youtube=youtube, channel_id=channel_id, start_date=start_date, end_date=end_date)

    

    transcripts = get_transcripts(video_urls=video_urls)
    return transcripts