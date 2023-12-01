import os
import spotipy
import spotipy.util as util
from typing import Optional
from pathlib import Path
from datetime import timedelta

import cohere
from dotenv import load_dotenv
from sklearn.cluster import KMeans

from user import User
from song import DownloadedSong, GeneratedSong, SongMeta
from embedding import generate_embedding_string
from get_lyrics_embedding import get_lyrics_embeddings
from find_best_cluster import find_best_cluster
from get_k_songs_closes_to_centroid import get_k_songs_closes_to_centroid
from ordenate_musics import ordenate_music
from save_previews import save_preview
from ffmpeg import merge
from utils import discretize
from get_lyrics import get_lyrics
from audio_features import *

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REDIRECT_URI = "http://localhost:8888/callback/"
SCOPE = "user-top-read"
N_TOP_SONGS = 30
N_CLUSTERS = 2

def fetch_meta(sp: spotipy.Spotify, user_id: str, song_id: str) -> Optional[SongMeta]:
    track_response = sp.track(track_id=song_id)

    preview_url = track_response["preview_url"]
    if preview_url is None:
        return None

    song_name = track_response["name"]
    song_main_artist=track_response["artists"][0]["name"]
    lyrics = get_lyrics(song_name, song_main_artist)

    if lyrics is None:
        return None

    features_response=sp.audio_features([song_id])[0]

    artist_id = track_response["artists"][0]["id"]
    artist_info = sp.artist(artist_id)

    genres: list[str] = artist_info["genres"]

    return SongMeta(
        id=song_id,
        song_name=song_name,
        song_main_artist=song_main_artist,
        lyrics=lyrics,
        danceability=Danceability(discretize(range=(0, 1), n_bins=5, value=features_response["danceability"])),
        energy=Energy(discretize(range=(0, 1), n_bins=5, value=features_response["energy"])),
        loudness=Loudness(discretize(range=(-60, 0), n_bins=3, value=features_response["loudness"])),
        speechiness=Speechiness(discretize(range=(0, 1), n_bins=3, value=features_response["speechiness"])),
        instrumentalness=Instrumentalness(discretize(range=(0, 1), n_bins=2, value=features_response["instrumentalness"])),
        valence=Valence(discretize(range=(0, 1), n_bins=5, value=features_response["valence"])),
        acousticness=Acousticness(discretize(range=(0, 1), n_bins=3, value=features_response["acousticness"])),
        song_bpm=Tempo(features_response["tempo"]),
        genres=genres,
        preview_url=preview_url
    )

def generate_song_from_user(accessToken: str, user_id: str, prompt: str) -> GeneratedSong:
    user = User(user_id)

    load_dotenv("../.env")
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    co = cohere.Client(os.getenv('COHERE_API_KEY'))
    sp = spotipy.Spotify(auth=accessToken)

    top_tracks_id = user.get_user_top_tracks(sp, N_TOP_SONGS)

    trackmeta_list: list[SongMeta] = []
    embedding_string_list = []
    for track_id in top_tracks_id:
        meta = fetch_meta(sp, user_id, track_id)
        if meta is None:
            logging.warn(f"skipping song")
            continue

        trackmeta_list.append(meta)
        embedding_string_list.append(generate_embedding_string(meta))

    embeddings = get_lyrics_embeddings(co, embedding_string_list)

    kmeans = KMeans(n_clusters=N_CLUSTERS)
    kmeans.fit(embeddings)

    best_cluster = find_best_cluster(co, kmeans, prompt)
    cluster_embeddings_idx = get_k_songs_closes_to_centroid(
        kmeans, best_cluster, embeddings, 4
    )
    tracks_ordered = ordenate_music(cluster_embeddings_idx, embeddings)

    downloaded_paths = []
    for track_idx in tracks_ordered:
        meta = trackmeta_list[track_idx]
        logging.info("chose song '%s' from '%s'", meta.song_name, meta.song_main_artist)
        downloaded_paths.append(save_preview(sp, user_id, meta))

    outpath = Path(f"data/{user_id}/out.mp3").absolute()
    merged_song = merge(downloaded_paths, outpath)
    return merged_song
