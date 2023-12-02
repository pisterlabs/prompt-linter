import os
import subprocess
import re
import random
import threading
import signal
import json
import time
from datetime import datetime, timezone, timedelta
import string

import openai

openai.api_key = ""

artists = {}
albums = {}
songs = {}

available_songs = []
used_songs = []
played_list = []

current_song = -1

duration_re = re.compile(r"\d+\.\d+")

music_proc = None

start_time = None
resume_point = 0
shuffle = True
loop = True

music_thread = None

music_end_condition = threading.Event()
music_stop_condition = threading.Event()
music_start_condition = threading.Event()

def play_sound(file_name):
    global start_time, music_proc, resume_point, music_stop_condition

    args = ["ffplay", "-autoexit", "-nodisp", "-ss", str(resume_point), file_name]
    music_proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    start_time = datetime.now(timezone.utc)

    out, err = music_proc.communicate()
    is_finished = music_proc.wait()

    if is_finished == 0:
        time.sleep(1)
        resume_point = 0
        music_stop_condition.set()


def initialize_all_music():
    artist_list = next(os.walk("./music/library"))[1]
    
    for artist in artist_list:
        tags = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": f"Create a list of tags for the band {artist}, seperated by ', '"}])

        similar_bands = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": f"{artist_list}"},
            {"role": "system", "content": f"Create a list of bands from the provided list that are similar to {artist}, seperated by ', '"}    
        ])

        artists[artist] = {
            "name": artist,
            "albums": [],
            "songs": [],
            "tags": tags.choices[0].message.content.split(", "),
            "similar_to": similar_bands.choices[0].message.content.split(", ")
        }

        album_list = next(os.walk(f"./music/library/{artist}"))[1]
        for album in album_list:
            artists[artist]["albums"].append(album)

            albums[album] = {
                "name": album,
                "artist": artist,
                "songs": []
            }
            song_list = next(os.walk(f"./music/library/{artist}/{album}"))[2]

            for song in song_list:
                rand_end = "<" + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + ">"

                artists[artist]["songs"].append(song.replace(".mp3", rand_end))
                albums[album]["songs"].append(song.replace(".mp3", rand_end))
            
                args=("ffprobe","-show_entries", "format=duration","-i",f"./music/library/{artist}/{album}/{song}")
                popen = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read()
                
                songs[song.replace(".mp3", rand_end)] = {
                    "name": song.replace(".mp3", ""),
                    "album": album,
                    "artist": artist,
                    "link": f"./music/library/{artist}/{album}/{song}",
                    "duration": duration_re.search(str(output)).group(0)
                    # "duration": str(output).replace("\\r\\n[/FORMAT]\\r\\n'", "").replace("b'[FORMAT]\\r\\nduration=", "")
                }

    with open("music/artists.json", 'w') as fp:
        json.dump(artists, fp)

    with open("music/albums.json", 'w') as fp:
        json.dump(albums, fp)

    with open("music/songs.json", 'w') as fp:
        json.dump(songs, fp)

def load_music():
    global artists, albums, songs
    artists = json.load(open("music/artists.json"))
    albums = json.load(open("music/albums.json"))
    songs = json.load(open("music/songs.json"))


def pick_new_song():
    global current_song
    
    picked_song = 0
    if shuffle:
        picked_song = random.randint(0, len(available_songs) - 1)
        # Reduces the chances of the same artist being picked twice in a row slightly
        if len(played_list) > 0 and songs[played_list[current_song]]["artist"] == songs[available_songs[picked_song]]["artist"]:
            picked_song = random.randint(0, len(available_songs) - 1)
    
    current_song += 1

    used_songs.append(available_songs[picked_song])
    played_list.append(available_songs[picked_song])
    available_songs.pop(picked_song)

    if( len(played_list) > (len(available_songs) + len(used_songs) )):
        played_list.pop(0)
        current_song -= 1

    if loop and len(used_songs) > 0.2 * (len(available_songs) + len(used_songs)):
        available_songs.append(used_songs.pop(0))

    print(f"{songs[played_list[current_song]]['artist']}/{songs[played_list[current_song]]['album']}/{songs[played_list[current_song]]['name']}")


def next_song(num):
    global current_song, music_proc, resume_point
    
    if current_song + num >= len(played_list):
        num_to_add = current_song - (len(played_list) - 1) + num
        for i in range(num_to_add):
            pick_new_song()

        current_song = len(played_list) - 1
    else:
        current_song += num

    resume_point = 0

    if music_proc:
        music_proc.kill()
        music_stop_condition.set()

    unpause()


def unpause():
    global music_start_condition
    music_start_condition.set()


def pause():
    global start_time, resume_point, music_proc, music_stop_condition

    if start_time:
        playback_time = datetime.now(timezone.utc) - start_time
        resume_point += playback_time.total_seconds()

        if music_proc:
            music_proc.kill()
        music_stop_condition.set()


def start_music_thread(should_shuffle=True, should_loop=True, album=None, artist=None, playlist=None):
    global current_song, shuffle, loop, music_stop_condition, music_start_condition, available_songs, used_songs

    available_songs = []
    used_songs = []

    shuffle = should_shuffle
    loop = should_loop

    if album != None:
        for song in albums[album]["songs"]:
            available_songs.append(song)
    elif artist != None:
        for song in artists[artist]["songs"]:
            available_songs.append(song)
    else:
        for song in songs:
            available_songs.append(song)
    
    pick_new_song()

    music_start_condition.wait()
    music_start_condition.clear()

    while True:
        threading.Thread(target=play_sound, args=(songs[played_list[current_song]]["link"],)).start()

        music_stop_condition.wait()
        music_stop_condition.clear()

        if music_end_condition.is_set():
            music_end_condition.clear()
            return

        if resume_point > 0:
            music_start_condition.wait()
            music_start_condition.clear()
        else:
            pick_new_song()

def end_music_thread():
    global music_stop_condition, music_end_condition
    music_proc.kill()
    music_end_condition.set()
    music_stop_condition.set()

def start_music(should_shuffle=True, should_loop=True, album=None, artist=None, playlist=None):
    global music_thread

    if music_thread:
        end_music_thread()

    music_thread = threading.Thread(target=start_music_thread, args=(should_shuffle, should_loop, album, artist, playlist))
    music_thread.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_music_command(command):
    command = command.split(" ")

    if  "pause" in command or "stop" in command:
        pause()
        return f"Music paused. Current song: {songs[played_list[current_song]]['artist']}-{songs[played_list[current_song]]['name']}"
    elif "unpause" in command or "resume" in command or "play" in command:
        unpause()
        return f"Music resumed. Current song: {songs[played_list[current_song]]['artist']}-{songs[played_list[current_song]]['name']}"
    elif "rewind" in command or "restart" in command:
        next_song(0)
        return f"Song rewound. Current song: {songs[played_list[current_song]]['artist']}-{songs[played_list[current_song]]['name']}"
    elif "back" in command or "previous" in command:
        num = -1
        for word in command:
            if word.isdigit():
                num = -1 * int(word)
        next_song(num)
        return f"Went back {num} songs. Current song: {songs[played_list[current_song]]['artist']}-{songs[played_list[current_song]]['name']}"
    elif "next" in command or "forward" in command or "skip" in command:
        num = 1
        for word in command:
            if word.isdigit():
                num = int(word)
        next_song(num)
        return f"Skipped ahead {num} songs. Current song: {songs[played_list[current_song]]['artist']}-{songs[played_list[current_song]]['name']}"

    return f"Unknown command: {command}"
    
# initialize_all_music()

load_music()
start_music()