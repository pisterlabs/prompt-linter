import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

# Load the API key from a file
with open("key.txt", "r") as key_file:
    api_key = key_file.read().strip()

openai.api_key = api_key

# Define the user message
username = input("Enter Spotify Username: ")
book_name = input("Enter Book Name/Movie Title: ")
author_name = input("Enter Author/Movie Director: ")
book_input = f"based on the synopsis, country of origin, language, and vibe of the book, Please provide a filled-out JSON file in this format {{\n  \"playlist_name\": \"\",\n  \"playlist_description\": \"\",\n  \"playlist_songs\": [\"\"]\n}}\nWith 10 songs from the last 50 years that fit the vibe of the book/movie \"{book_name}\" by {author_name} for the spotify api. Do not respond with anything other than the JSON file. If you can include songs from the soundtrack. The songs should be relevant to the intended audience, setting of book, etc.,. Do not respond with anything other than the JSON file."
user_message = {"role": "user", "content": book_input}

# Create a chat completion request
completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[user_message]
)

# Extract and parse the assistant's reply as JSON
assistant_reply = completion['choices'][0]['message']['content']
playlist_info = json.loads(assistant_reply)

# Spotify API credentials
client_id = '0a88ea0d18d44e1da084a214dc9fd9c4'
client_secret = "6d8b25bf74e04a9d8e11d6389bc55b0d"
redirect_uri = "http://127.0.0.1:8080/"

# User-specific settings
scope = 'playlist-modify-public'

try:
    # Authenticate with Spotify
    token = SpotifyOAuth(scope=scope, username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    spotifyObject = spotipy.Spotify(auth_manager=token)

    # Create a new playlist
    playlist_name = playlist_info['playlist_name']
    playlist_description = playlist_info['playlist_description']

    playlist = spotifyObject.user_playlist_create(user=username, name=playlist_name, public=True, description=playlist_description)
    print(f"Playlist '{playlist_name}' created successfully!")

    # Add songs to the playlist
    list_of_songs = []
    for song in playlist_info['playlist_songs']:
        result = spotifyObject.search(q=song)
        if result['tracks']['items']:
            list_of_songs.append(result['tracks']['items'][0]['uri'])

    # Find the new playlist
    prePlaylist = spotifyObject.user_playlists(user=username)
    playlist_id = prePlaylist['items'][0]['id']

    # Add songs to the playlist
    if list_of_songs:
        spotifyObject.user_playlist_add_tracks(user=username, playlist_id=playlist_id, tracks=list_of_songs)
        print(f"Songs added to the playlist '{playlist_name}' successfully!")
    else:
        print("No songs were added to the playlist.")
except Exception as e:
    print(f"Error: {e}")
