from flask import Flask, request, abort, jsonify, session, url_for, redirect
from flask_bcrypt import Bcrypt
import os
import openai
from image_functions import compress_image, get_dalle_image
from dotenv import load_dotenv
from models import db, User
from config import ApplicationConfig
from flask_session import Session
from flask_cors import CORS, cross_origin
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import base64
import requests
import json

load_dotenv()

spotify_client_id = os.environ["SPOTIFY_CLIENT_ID"]
spotify_client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
gpt_key = os.environ["gpt_key"]



def get_songs_gpt(numSongs, musicVibe, musicGenre,artistName,musicDecade, musicType):
    openai.api_key = gpt_key

    content = f"""Hello! I want a list of {numSongs} with a {musicVibe} vibe. 
    The songs should have the style from the decade {musicDecade} and be of the genre {musicGenre}.
    I want the songs to be {musicType} and have at least one song by {artistName}
    each song in the list is represented by a number followed by a period, 
    the song name in single quotes, and the artist name after the "by" keyword"""
   
    message = [
    {"role": "user", "content": content}
    ]

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=.8,
    messages= message
    )


    gptmessage = completion.choices[0].message.content

    return parse_songs_to_dic(gptmessage)

def parse_songs_to_dic(gptmessage):
    song_dict = {}
    lines = gptmessage.split('\n')  # split the message by line
    for line in lines:
        if line[0] in '123456789': #check if line is a song line
          song_name, artist = line.split(" by ") #split line
          song_name = song_name[4:-1] #trim song name
          song_dict[song_name] = artist #assign to dict

    return song_dict



#SPOTIFY OAUTH
def create_spotify_oauth():
    return SpotifyOAuth(
        client_id = spotify_client_id,
        client_secret=spotify_client_secret,
        redirect_uri=url_for('redirectPage', _external=True),
        scope='user-library-read,user-library-modify,playlist-modify-private,ugc-image-upload'
    )

def get_track_id(token, track, artist):
    sp_oauth = create_spotify_oauth()
    code = token
    track_name = track
    artist_name = artist
    token_info = sp_oauth.get_access_token(code)
    spotify_object = spotipy.Spotify(auth=token_info['access_token'])
    track_query = spotify_object.search(q=f'track:{track_name} artist:{artist_name}')
    track_id = track_query['tracks']['items'][0]['uri']
    return track_id

def create_playlist(token, name):
    sp_oauth = create_spotify_oauth()
    code = token
    playlist_name = name
    token_info = sp_oauth.get_access_token(code)
    spotify_object = spotipy.Spotify(auth=token_info['access_token'])
    user_id = spotify_object.current_user()['id'] #Spotify UID of current user
    create_playlist = spotify_object.user_playlist_create(user_id,playlist_name,False,False,'An Ai Generated Playlist')
    return create_playlist['id'] #returns ID of created playlist

def update_playlist_cover(token, playlist_id, image_url):
    sp_oauth = create_spotify_oauth()
    code = token
    token_info = sp_oauth.get_access_token(code)
    spotify_object = spotipy.Spotify(auth=token_info['access_token'])
    base64_image = compress_image(image_url, 60)
    update_picture = spotify_object.playlist_upload_cover_image(playlist_id, base64_image)
    return update_picture


def add_tracks_to_playlist(token, playlist_id, track_list):#track_list is a list of song IDs
    sp_oauth = create_spotify_oauth()
    code = token
    token_info = sp_oauth.get_access_token(code)
    spotify_object = spotipy.Spotify(auth=token_info['access_token'])
    add_to_playlist = spotify_object.playlist_add_items(playlist_id,track_list)
    return add_to_playlist 
