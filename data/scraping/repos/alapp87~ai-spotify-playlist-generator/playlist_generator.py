import os
import json
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

import openai
import spotipy


log = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

client_id = None
client_secret = None


def main():
    setup_environment()

    args = parse_args()

    playlist_desc = args.d
    playlist_name = args.p or generate_playlist_name()
    playlist_length = args.n

    sp = login_to_spotify()

    current_user = sp.current_user()
    assert current_user != None

    log.info("Logged into Spotify")

    log.info("Generating songs for playlist...")

    songs = generate_song_playlist(playlist_desc, playlist_length)
    log.info("Got songs for playlist")

    track_ids = []

    for item in songs:
        artist, song = item["artist"], item["song"]
        track_id = search_for_song(sp, artist, song)
        if track_id:
            track_ids.append(track_id)

    log.info("Creating playlist...")
    playlist = create_playlist(sp, playlist_name)
    add_tracks_to_playlist(sp, current_user["id"], playlist, track_ids)

    log.info(
        'Generated new playlist "%s" with %s songs!', playlist_name, playlist_length
    )


def setup_environment() -> None:
    global client_id, client_secret

    load_dotenv(".env")

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Spotify playlist using ChatGPT"
    )
    parser.add_argument(
        "-d", type=str, help="The description to generate the playlist", required=True
    )
    parser.add_argument("-p", type=str, help="The playlist name", required=False)
    parser.add_argument("-n", type=int, help="Number of songs in playlist", default=10)

    return parser.parse_args()


def generate_playlist_name() -> str:
    date_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    return f"My Playlist ({date_str})"


def generate_song_playlist(prompt, count):
    SYSTEM_PROMPT = """
    You are a helpful playlist generating assistant. You should generate a list of
    songs and their artists according to a text prompt. You should return a JSON array
    where each element follows this format: {"song": <song_title>, "artist": <artist_name>}
    """

    example_response = """
    [
       {"song": "Hurt", "artist": "Johnny Cash"},
       {"song": "Someone Like You", "artist": "Adele"},
       {"song": "Fix You", "artist": "Coldplay"},
       {"song": "Nothing Compares 2 U", "artist": "Sinead O'Connor"},
       {"song": "Tears in Heaven", "artist": "Eric Clapton"},
       {"song": "Yesterday", "artist": "The Beatles"},
       {"song": "Everybody Hurts", "artist": "R.E.M."},
       {"song": "Mad World", "artist": "Gary Jules"},
       {"song": "Skinny Love", "artist": "Bon Iver"},
       {"song": "Creep", "artist": "Radiohead"}
    ]
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Generate a playlist of 10 songs based on this prompt: super super sad songs",
        },
        {"role": "assistant", "content": example_response},
        {
            "role": "user",
            "content": f"Generate a playlist of {count} songs based on this prompt: {prompt}",
        },
    ]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=300
    )

    return json.loads(res["choices"][0]["message"]["content"])


def login_to_spotify():
    return spotipy.Spotify(
        auth_manager=spotipy.SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://localhost:9999/",
            scope="playlist-modify-private",
        )
    )


def search_for_song(sp, artist, song):
    advanced_query = f"artist:({artist}) track:({song})"
    basic_query = f"{song} {artist}"

    for query in [advanced_query, basic_query]:
        log.debug("Using query to find song: %s", query)
        search_results = sp.search(q=query, type="track", limit=1)
        if (
            not search_results["tracks"]["items"]
            or search_results["tracks"]["items"][0]["popularity"] < 20
        ):
            continue
        else:
            good_guess = search_results["tracks"]["items"][0]
            name = good_guess["name"]
            id = good_guess["id"]
            log.debug("Found: %s [%s]", name, id)

            return id
    else:
        log.info(
            "Queries %s and %s returned no good results. Skipping.",
            advanced_query,
            basic_query,
        )

    return None


def create_playlist(sp: spotipy.Spotify, playlist_name: str):
    user_id = sp.current_user()["id"]
    return sp.user_playlist_create(user_id, public=False, name=playlist_name)


def add_tracks_to_playlist(
    sp: spotipy.Spotify, user_id: str, playlist, track_ids: list[str]
):
    sp.user_playlist_add_tracks(user_id, playlist["id"], track_ids)


if __name__ == "__main__":
    main()
