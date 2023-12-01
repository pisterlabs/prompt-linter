from googleapiclient.discovery import build
import openai
import main as m
import pandas as pd
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Read the GPT-3 API key from a file
def read_api_key(filename):
    with open(filename, 'r') as file:
        gpt_api_key = file.readline().strip()
    return gpt_api_key

# Set the GPT-3 API key
openai.api_key = read_api_key("gpt_api_key.txt")

# Define the initial system message for the conversation
messages = [
    {
        "role": "system",
        "content": "You are a personal assistant that analyzes YouTube analytics. You shall analyze the YouTube analytics of a channel. "
                    "I shall provide you with the channel name, description, subscriberCount, views, videoCount, and region. "
                    "Keep a professional tone while describing the channel."
                    "I shall provide you with the top 10 videos of the channel. "
                    "Make the report engaging and interesting, covering most aspects."
    }
]

# Read the YouTube API key from a file
with open('yt_key.txt', 'r') as f:
    api_key = f.read()

# Read the YouTube API key for the channel you want to analyze
youtube_api_key = read_api_key("yt_key.txt")
#channel_id = 'UCX6OQ3DkcsbYNE6H8uQQuVA'


def get_channel_id(api_key, channel_name):
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

    try:
        # Search for the channel by name
        search_response = youtube.search().list(
            q=channel_name,
            type='channel',
            part='id'
        ).execute()

        # Extract the channel ID from the search results
        if 'items' in search_response:
            channel = search_response['items'][0]
            channel_id = channel['id']['channelId']
            return channel_id

    except HttpError as e:
        print(f"An error occurred: {e}")
    
    return None

channel_id = get_channel_id(youtube_api_key, input("Enter the channel name: "))



# Build the YouTube API service
youtube = build('youtube', 'v3', developerKey=api_key)

# Get channel statistics using the 'get_channel_statistics' function from 'main.py'
data = m.get_channel_statistics(youtube, channel_id)

# Append user messages with channel details to the 'messages' list
messages.append({"role": "user", "content": "The channel name is " + data['Channel_name'] + "."})
messages.append({"role": "user", "content": "The channel description is " + data['description'] + "."})
messages.append({"role": "user", "content": "The channel has " + data['Subscribers'] + " subscribers."})
messages.append({"role": "user", "content": "The channel has " + data['Views'] + " views."})
messages.append({"role": "user", "content": "The channel has " + data['Total_Videos'] + " videos."})
messages.append({"role": "user", "content": "The channel is from " + data['country'] + "."})

# Create a DataFrame for channel data
channel_data = pd.DataFrame([data])

# Drop the 'description' and 'country' columns
channel_data = channel_data.drop(columns=['description', 'country'])

# Set the correct data types for each column
channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
channel_data['Views'] = pd.to_numeric(channel_data['Views'])
channel_data['Total_Videos'] = pd.to_numeric(channel_data['Total_Videos'])

# Get video details using the 'get_channel_videos' and 'get_video_details' functions from 'main.py'
video_ids = m.get_channel_videos(youtube, channel_data['playlist_id'][0])
video_details = m.get_video_details(youtube, video_ids)

# Create a DataFrame for video data
video_data = pd.DataFrame(video_details)

# Convert 'publishedAt', 'viewCount', 'likeCount', and 'dislikeCount' to the correct data types
video_data['publishedAt'] = pd.to_datetime(video_data['publishedAt'])
video_data['viewCount'] = pd.to_numeric(video_data['viewCount'])
video_data['likeCount'] = pd.to_numeric(video_data['likeCount'])
video_data['dislikeCount'] = pd.to_numeric(video_data['dislikeCount'])

# Sort the video_data DataFrame to get the top 10 videos by view count
top_10 = video_data.sort_values(by='viewCount', ascending=False).head(10)

# Loop through the top 10 videos and add user messages with video details to the 'messages' list
for index, row in top_10.iterrows():
    video_title = row['Title']
    video_views = row['viewCount']
    video_likes = row['likeCount']
    video_dislikes = row['dislikeCount']
    
    # Create a user message with video details
    user_message = {
        "role": "user",
        "content": f"The video '{video_title}' has {video_views} views, {video_likes} likes, and {video_dislikes} dislikes. "
                    "This is a part of their top 10 videos."
    }
    
    # Append the user message to the 'messages' list
    messages.append(user_message)

# Use ChatGPT to generate a response based on the conversation
chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.4,
    max_tokens=1500
)

# Get the generated reply from ChatGPT
reply = chat.choices[0].message.content
print(reply)



#write the reply into a text file
with open('reply.txt', 'w') as f:
    f.write(reply)