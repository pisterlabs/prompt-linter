import argparse
import datetime
import logging
import os
import json

import openai
import spotipy
from dotenv import load_dotenv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Simple command line utility")
    parser.add_argument("-p", type=str, help="The prompt to describing the playlist.")
    parser.add_argument("-n", type=int, default="12", help="The number of songs to be added.")
    parser.add_argument("-envfile", type=str, default="D:\GPTProjects\.env", required=False, help='A dotenv file with your environment variables: "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "OPENAI_API_KEY"')

    args = parser.parse_args()
    load_dotenv(args.envfile)
    if any([x not in os.environ for x in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "OPENAI_API_KEY")]):
        raise ValueError("Error: missing environment variables. Please check your env file.")
    if args.n not in range(1,50):
        raise ValueError("Error: n should be between 0 and 50")

    openai.api_key = os.environ["OPENAI_API_KEY"]

    playlist_prompt = args.p
    count = args.n
    playlist = get_playlist(playlist_prompt, count)
    add_songs_to_spotify(playlist_prompt, playlist)

def get_playlist(prompt, count=8):
    example_json = """
    [
      {"song": "Everybody Hurts", "artist": "R.E.M."},
      {"song": "Nothing Compares 2 U", "artist": "Sinead O'Connor"},
      {"song": "Tears in Heaven", "artist": "Eric Clapton"},
      {"song": "Hurt", "artist": "Johnny Cash"},
      {"song": "Yesterday", "artist": "The Beatles"}
    ]
    """
    messages = [
        {"role": "system", "content": """You are a helpful playlist generating assistant. 
        You should generate a list of songs and their artists according to a text prompt.
        Your should return a JSON array, where each element follows this format: {"song": <song_title>, "artist": <artist_name>}
        """
        },
        {"role": "user", "content": "Generate a playlist of 5 songs based on this prompt: super super sad songs"},
        {"role": "assistant", "content": example_json},
        {"role": "user", "content": f"Generate a playlist of {count} songs based on this prompt: {prompt}"},
    ]

    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=400
    )

    playlist = json.loads(response["choices"][0]["message"]["content"])
    return playlist


def add_songs_to_spotify(playlist_prompt, playlist):
    # Sign up as a developer and register your app at https://developer.spotify.com/dashboard/applications

    # Step 1. Create an Application.

    # Step 2. Copy your Client ID and Client Secret.
    spotipy_client_id = os.environ["SPOTIFY_CLIENT_ID"]  # Use your Spotify API's keypair's Client ID
    spotipy_client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]  # Use your Spotify API's keypair's Client Secret

    # Step 3. Click `Edit Settings`, add `http://localhost:9999` as as a "Redirect URI"
    spotipy_redirect_url = "http://localhost:9999"  # Your browser will return page not found at this step. We'll grab the URL and paste back in to our console

    # Step 4. Click `Users and Access`. Add your Spotify account to the list of users (identified by your email address)

    # Spotipy Documentation
    # https://spotipy.readthedocs.io/en/2.22.1/#getting-started

    sp = spotipy.Spotify(
        auth_manager=spotipy.SpotifyOAuth(
            client_id=spotipy_client_id,
            client_secret=spotipy_client_secret,
            redirect_uri=spotipy_redirect_url,
            scope="playlist-modify-private",
        )
    )
    current_user = sp.current_user()

    assert current_user is not None

    track_uris = []
    for item in playlist:
        artist, song = item["artist"], item["song"]
        # https://developer.spotify.com/documentation/web-api/reference/#/operations/search

        advanced_query = f"artist:({artist}) track:({song})"
        basic_query = f"{song} {artist}"

        for query in [advanced_query, basic_query]:
            log.debug(f"Searching for query: {query}")
            search_results = sp.search(q=query, limit=10, type="track")  # , market=market)

            if not search_results["tracks"]["items"] or search_results["tracks"]["items"][0]["popularity"] < 20:
                continue
            else:
                good_guess = search_results["tracks"]["items"][0]
                print(f"Found: {good_guess['name']} [{good_guess['id']}]")
                # print(f"FOUND USING QUERY: {query}")
                track_uris.append(good_guess["id"])
                break

        else:
            print(f"Queries {advanced_query} and {basic_query} returned no good results. Skipping.")

    created_playlist = sp.user_playlist_create(
        current_user["id"],
        public=False,
        name=f"{playlist_prompt} ({datetime.datetime.now().strftime('%c')})",
    )

    sp.user_playlist_add_tracks(current_user["id"], created_playlist["id"], track_uris)

    print("\n")
    print(f"Created playlist: {created_playlist['name']}")
    print(created_playlist["external_urls"]["spotify"])


if __name__ == "__main__":
    main()
