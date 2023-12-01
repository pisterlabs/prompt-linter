import json
import os
import subprocess

import openai
import spotipy
from termcolor import colored

import sp_auth
from config import ACCESS_TOKEN_FILE, DATA_PATH
from playlists_db import PlaylistManager


def play_playlist(sp, tracks_list, device_id, playlist_name):
    """
    Play a playlist on the given device using the Spotify API.
    If the given 'playlist_name' does not exist in the user's playlists, it will be created.
    if it does exist, it will just be played.
    :param sp:
    :param tracks_list: the playlist json
    :param device_id:
    :param playlist_name: the name of the playlist the user gave
    """
    # Get the user's playlists
    playlists = sp.current_user_playlists()

    # Check if playlist with the given name already exists
    playlist_id = None
    for item in playlists['items']:
        if item['name'] == playlist_name:
            playlist_id = item['id']
            break

    user_id = sp.me()['id']

    # If no existing playlist is found, create a new one
    if playlist_id is None:
        track_uris = []
        for track in tracks_list["playlist"]:
            results = sp.search(q='track:{} artist:{}'.format(track['song_name'], track['artist_name']), type='track')
            if results['tracks']['items']:
                track_uris.append(results['tracks']['items'][0]['uri'])
        print(colored(f"Creating playlist {playlist_name} with {len(track_uris)} tracks", "yellow"))
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name)
        playlist_id = playlist['id']
        sp.user_playlist_replace_tracks(user=user_id, playlist_id=playlist_id, tracks=track_uris)

    # Start playback on the selected device for the given playlist
    sp.start_playback(device_id=device_id, context_uri=f'spotify:playlist:{playlist_id}')


def generate_tracks_list(playlist_description) -> json:
    """
    Generate a list of tracks based on the user's 'playlist_description' using GPT
    :param playlist_description:
    :return: a JSON describing the tracks list
    """
    # Get the OpenAI API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Set the OpenAI API key
    openai.api_key = api_key

    # Construct the prompt for GPT-4
    prompt = f"""
{playlist_description}. Generate a list of 15 songs in the format of a JSON with song name and artist name:"
Use the following JSON format:
{{
    "playlist":
    [
        {{"song_name": "The long and winding road", "artist_name": "The Beatles"}},
        {{"song_name": "Sweet Child o' Mine", "artist_name": "Guns N' Roses"}},
    ]
}}
"""

    # Call the GPT-4 model to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",  # Set the model
        messages=[
            {"role": "system", "content": "You are a knowledgeable AI trained to generate music playlists."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the assistant's reply (assumes the reply is the last message)
    assistant_reply = response['choices'][0]['message']['content']

    # Parse JSON and return it
    playlist = json.loads(assistant_reply)

    return playlist


def setup_spotify():
    # Read access token from creds/access_token.txt
    # to generate this file, run sp_auth.py
    with open(ACCESS_TOKEN_FILE, "r") as f:
        access_token = f.read()
        return spotipy.Spotify(auth=access_token)


def authorize_spotify():
    sp_auth.run_flow()


def main():
    # if token file does not exist
    if not os.path.exists(ACCESS_TOKEN_FILE):
        print(colored("Running authorization flow", "red", attrs=["bold"]))
        authorize_spotify()
        exit(1)
    sp = setup_spotify()

    # Ask the user for their desired playlist
    pm = PlaylistManager(DATA_PATH)
    print(colored("Here are your playlists:", "green"))
    playlists = pm.list_playlists()
    for i, playlist in enumerate(playlists, 0):
        print(f"{i}. {playlist}")

    playlist_description = input(
        colored("\nEnter an playlist number OR a description for a new playlist you want:\n", "green", attrs=["bold"]))

    print(colored("Opening your spotify desktop app", "yellow", attrs=["bold"]))
    command = "/Applications/Spotify.app/Contents/MacOS/Spotify"
    subprocess.Popen(command)

    if playlist_description.isdigit():
        # Load old playlist
        playlist = pm.load_playlist(int(playlist_description))
        playlist_description = pm.get_playlist_name(int(playlist_description))
        print(colored(f"Loading {playlist_description}...", "yellow", attrs=["bold"]))
    else:
        # Generate new playlist
        print(colored("Generating playlist...", "yellow", attrs=["bold"]))
        playlist = generate_tracks_list(playlist_description)
        pm.save_playlist(playlist_description, playlist)

    print(colored("Playing:", "green"))
    text_list = playlist_json_to_text(playlist)
    print(colored(text_list, "yellow"))

    try:
        devices = sp.devices()
        device_id = devices['devices'][0]['id']  # get the first device

        print(colored("\n\nPlaying...", "yellow", attrs=["bold"]))

        play_playlist(sp, playlist, device_id, playlist_description)
    except spotipy.exceptions.SpotifyException:
        print(colored("Your spotify token has expired, running authorization flow", "red", attrs=["bold"]))
        authorize_spotify()


def playlist_json_to_text(playlist):
    text_list = ""
    for i, song in enumerate(playlist["playlist"], start=1):
        text_list += f"{i}. {song['song_name']} by {song['artist_name']}\n"
    return text_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
