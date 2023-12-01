# spotify authentication
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotipy.exceptions import SpotifyException
# language model
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage  
# harvesting track information
import re
import lyricsgenius
import requests
from bs4 import BeautifulSoup
# comparing genres
from sentence_transformers import SentenceTransformer
import pprint
from sklearn.metrics.pairwise import cosine_similarity
import os
from fuzzywuzzy import fuzz
import random
# loading API keys
from dotenv import load_dotenv
load_dotenv() 


#llm = ChatOpenAI(max_retries=3, temperature=0, model_name = "gpt-4")

# let's try using one central llm [error: circular import]
#from main import llm

### ### ### Spotify Setup ### ### ###


scope= ['user-library-read', 
        'user-read-playback-state', 
        'user-modify-playback-state'] 


sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(), 
    auth_manager=SpotifyOAuth(scope=scope)
)


devices = sp.devices()
device_id = devices['devices'][0]['id']


### ### ### Basic Functions ### ### ###


def find_track_by_name(track_name):
    """
    Searches track by name to get uri
    """
    results = sp.search(q=track_name, type='track')
    track_uri = results['tracks']['items'][0]['uri']
    return track_uri


def play_track_by_name(track_name): 
    """
    Plays track given its name
    For less popular tracks, do 'play {track} {artist}'
    """
    track_uri = find_track_by_name(track_name)
    track_name = sp.track(track_uri)["name"]
    artist_name = sp.track(track_uri)['artists'][0]['name']
    try:
        sp.start_playback(device_id=device_id, uris=[track_uri])
        return f"♫ Now playing {track_name} by {artist_name} ♫"
    except SpotifyException as e:
        return f"An error occurred with Spotify: {e}. \n\n**Remember to wake up Spotify.**"
    except Exception as e: 
        return f"An unexpected error occurred: {e}."


def queue_track_by_name(track_name):
    """
    Queues track given its name
    """
    track_uri = find_track_by_name(track_name)
    track_name = sp.track(track_uri)["name"]
    sp.add_to_queue(uri=track_uri, device_id=device_id)
    return f"♫ Added {track_name} to your queue ♫"
  

def pause_track():
    sp.pause_playback(device_id=device_id)
    return "♫ Playback paused ♫"


def play_track():
    sp.start_playback(device_id=device_id)
    return "♫ Playback started ♫"


def skip_track():
    sp.next_track(device_id=device_id)
    return "♫ Skipped to your next track ♫"
 
    
### ### ### More Elaborate ### ### ###
    

def play_album_by_name_and_artist(album_name, artist_name):
    """
    Plays an album given its name and artist
    context_uri (for artist, album, or playlist) expects a string
    """
    results = sp.search(q=f'{album_name} {artist_name}', type='album')
    album_id = results['albums']['items'][0]['id']
    album_info = sp.album(album_id)
    album_name = album_info['name']
    artist_name = album_info['artists'][0]['name']
    try: 
        sp.start_playback(device_id=device_id, context_uri=f'spotify:album:{album_id}')
        return f"♫ Now playing {album_name} by {artist_name} ♫"
    except spotipy.SpotifyException as e:
        return f"An error occurred with Spotify: {e}. \n\n**Remember to wake up Spotify.**"
    except Exception as e: 
        return f"An unexpected error occurred: {e}."


def play_playlist_by_name(playlist_name):
    from main import llm
    """
    This function takes a user-provided playlist name as input and plays 
    the name of the real, matching playlist from the user's existing playlists on Spotify.
    """
    playlists = sp.current_user_playlists()
    playlist_dict = {playlist['name']: (playlist['id'], playlist['owner']['display_name']) for playlist in playlists['items']}
    
    system_message_content = """
        The task is to find a match between the user-requested playlist name and
        the user's existing playlists. Only return the name of the match.
        """
    human_message_content = f"""
        User's requested playlist: {playlist_name}
        User's existing playlists: {list(playlist_dict.keys())}
        """
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]

    ai_response = llm(messages).content
    playlist_name = ai_response.strip("'")
    try: 
        playlist_id, creator_name = playlist_dict[playlist_name]
        sp.start_playback(device_id=device_id, context_uri=f'spotify:playlist:{playlist_id}')
        return f'♫ Now playing {playlist_name} by {creator_name} ♫'
    except: 
        return "Unable to find playlist. Please try again."


# avoid creating new object everytime function is called
genius = lyricsgenius.Genius() 
def get_track_info(): 
    """
    Harvests information for explain_track() using the Genius API and webscraping. 
    """
    current_track_item = sp.current_user_playing_track()['item']
    track_name = current_track_item['name']
    artist_name = current_track_item['artists'][0]['name']
    album_name = current_track_item['album']['name']
    release_date = current_track_item['album']['release_date']
    basic_info = {
        'track_name': track_name,
        'artist_name': artist_name,
        'album_name': album_name,
        'release_date': release_date,
    }
    # features hinder Genius search
    track_name = re.split(' \(with | \(feat\. ', track_name)[0]
    result = genius.search_song(track_name, artist_name)

    # if no Genius page exists
    spotify_artist = artist_name.lower().replace(" ", "")
    genius_artist = result.artist.lower().replace(" ", "")
    print(spotify_artist)
    print(genius_artist)
    if spotify_artist not in genius_artist:
        return basic_info, None, None, None
    
    # if a Genius page exists
    lyrics = result.lyrics
    url = result.url
    response = requests.get(url)

    # parsing the webpage & locating About section
    soup = BeautifulSoup(response.text, 'html.parser')
    about_section = soup.select_one('div[class^="RichText__Container-oz284w-0"]')

    # if no About section exists
    if not about_section: 
        return basic_info, None, lyrics, url
    
    # if an About section exists
    else: 
        about_section = about_section.get_text(separator='\n')
        return basic_info, about_section, lyrics, url


def explain_track(): 
    from main import llm
    """
    Presents track information in an organized, compelling manner. 
    """
    basic_info, about_section, lyrics, url = get_track_info()
    print(basic_info, about_section, lyrics, url)
    if lyrics: # (essentially, if a Genius page exists; lyrics != None)
        system_message_content = """
            Your task is to create an engaging summary for a track using the available details
            about the track and its lyrics. If there's insufficient or no additional information
            besides the lyrics, craft the entire summary based solely on the lyrical content."
            """
        human_message_content = f"{about_section}\n\n{lyrics}"
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=human_message_content)
        ]
        ai_response = llm(messages).content
        summary = f"""
            **Name:** <span style="color: red; font-weight: bold; font-style: italic;">{basic_info["track_name"]}</span>   
            **Artist:** {basic_info["artist_name"]}   
            **Album:** {basic_info["album_name"]}   
            **Release:** {basic_info["release_date"]}   

            **About:** 
            {ai_response}

            <a href='{url}'>Click here for more information on Genius!</a>  
        """
        return summary
    else: # (if no Genius page exists - Lyrics;  None)
        url = "https://genius.com/Genius-how-to-add-songs-to-genius-annotated"
        summary = f"""
            **Name:** <span style="color: red; font-weight: bold; font-style: italic;">{basic_info["track_name"]}</span>   
            **Artist:** {basic_info["artist_name"]}   
            **Album:** {basic_info["album_name"]}   
            **Release:** {basic_info["release_date"]}   

            **About:** 
            Unfortunately, this track has not been uploaded to Genius.com

            <a href='{url}'>Be the first to change that!</a>  
        """
        return summary



### ### ### Genre + Mood ### ### ###


def get_user_mood(user_mood):
    from main import llm
    """
    """
    system_message_content = """
        Your task is to evaluate the user's response to the question
        'How are you feeling today' and determine their emotional state. 
        Categorize this state into one of the four predefined moods: happy, sad, energetic, or calm. 
        Remember, always return only the mood category as your output.
        """
    human_message_content = user_mood
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]
    ai_response = llm(messages).content
    user_mood = ai_response
    return user_mood


os.environ["TOKENIZERS_PARALLELISM"] = "false" # satisfies warning
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"""
* MiniLM models are smaller and faster versions of BERT-like models with competitive performance. 
* L6 indicates the number of layers in the model -
a reduced number of layers compared to larger models like BERT, which typically have 12 or 24 layers. 
"""
genre_list = sp.recommendation_genre_seeds()["genres"] # genres accepted by recommendations()
#pprint.pprint(genre_list)
genre_embeddings = model.encode(genre_list)


def get_genre_by_name(genre_name): 
    """
    Matches genre_name input to closest existing genre in genre_list;
    recommendations() only accepts from list
    """
    if genre_name.lower() in genre_list:
        genre_name = genre_name.lower()
        return genre_name
    else:
        genre_name_embedding = model.encode([genre_name.lower()])
        similarity_scores = cosine_similarity(genre_name_embedding, genre_embeddings)
        most_similar_index = similarity_scores.argmax()
        genre_name = genre_list[most_similar_index]
        return genre_name
    

MOOD_SETTINGS = {
    "happy": {"max_instrumentalness": 0.001, "min_valence": 0.6},
    "sad": {"max_danceability": 0.65, "max_valence": 0.4},
    "energetic": {"min_tempo": 130, "min_danceability": 0.75},
    "calm": {"max_energy": 0.65, "max_tempo": 130}
}


from fuzzywuzzy import fuzz
similarity_threshold = 75 # found after testing; out of 100
def is_genre_match(genre1, genre2, threshold=similarity_threshold):
    """
    if you have a short string and a longer one, 
    fuzz.partial_token_set_ratio() will score the similarity
    of the shorter string to every possible sub-string
    (of the same length as the shorter string) in the longer string. 
    The highest similarity score is then returned.
    """
    score = fuzz.token_set_ratio(genre1, genre2)
    print(score) # debugging
    return score >= threshold


def create_track_list_str(track_uris):
    """
    Helper function; creates list of track names and corresponding artists
    """
    track_details = sp.tracks(track_uris)
    track_names_with_artists = [f"{track['name']} by {track['artists'][0]['name']}" for track in track_details['tracks']]
    track_list_str = "<br>".join(track_names_with_artists) 
    return track_list_str


NUM_ARTISTS = 20 # max 50
TIME_RANGE = "medium_term"
NUM_TRACKS = 10
MAX_ARTISTS = 4
def play_genre_by_name_and_mood(genre_name, user_mood):
    """
    1. Retrieves user's genre request and their current mood
    2. Matches genre and mood to existing options
    3. Gets 4 of user's top artists that align with genre
    4. Conducts personalized recommendations() search
    """
    genre_name = get_genre_by_name(genre_name)
    user_mood = get_user_mood(user_mood).lower()
    #mood_setting = mood_settings[user_mood]
    print(genre_name) 
    print(user_mood)
    #print(mood_setting)
    
    # increased personalization
    user_top_artists = sp.current_user_top_artists(limit=NUM_ARTISTS, time_range=TIME_RANGE) 
    matching_artists_ids = []

    ### READABLE ###
    for artist in user_top_artists['items']:
        #print(artist['genres']) 
        for artist_genre in artist['genres']:
            if is_genre_match(genre_name, artist_genre):
                matching_artists_ids.append(artist['id'])
                break # don't waste time cycling artist genres after match
        if len(matching_artists_ids) == MAX_ARTISTS: # limit to pass to recommendations()
            break 

    ### EFFICIENT ### [WORSE VERSION] ~ INCLUDES ARTIST MULTIPLE TIMES PER MATCH
    # matching_artists_ids = list(islice((artist['id'] for artist in user_top_artists['items'] 
    #                                for artist_genre in artist['genres']
    #                                if is_genre_match(genre_name, artist_genre)), MAX_ARTISTS))

    if not matching_artists_ids:
        matching_artists_ids = None
    else: 
        artist_names = [artist['name'] for artist in sp.artists(matching_artists_ids)['artists']]
        print(artist_names)
        print(matching_artists_ids)

    recommendations = sp.recommendations( # can pass maximum {genre + artists} = 5 seeds
                                    seed_artists=matching_artists_ids, 
                                    seed_genres=[genre_name], 
                                    seed_tracks=None, 
                                    limit=NUM_TRACKS, # number of tracks to return
                                    country=None,
                                    **MOOD_SETTINGS[user_mood])
                                    
    track_uris = [track['uri'] for track in recommendations['tracks']]
    track_list_str = create_track_list_str(track_uris)
    sp.start_playback(device_id=device_id, uris=track_uris)

    return f"""
    **♫ Now Playing:** <span style="color: red; font-weight: bold; font-style: italic;">{genre_name}</span> ♫

    **Selected Tracks:**
    {track_list_str}
    """


### ### ### Artist + Mood ### ### ###


NUM_ALBUMS = 20 # max 20
MAX_TRACKS = 10
def play_artist_by_name_and_mood(artist_name, user_mood):
    user_mood = get_user_mood(user_mood).lower()
    print(user_mood)

    # retrieving and shuffling all artist's tracks
    results = sp.search(q=artist_name, type='artist')
    artist_id = results['artists']['items'][0]['id']
    # most recent albums retrieved first
    artist_albums = sp.artist_albums(artist_id, album_type='album', limit=NUM_ALBUMS)
    artist_tracks = []
    for album in artist_albums['items']:
        album_tracks = sp.album_tracks(album['id'])['items']
        artist_tracks.extend(album_tracks)
    random.shuffle(artist_tracks) 

    # filtering until we find enough tracks that match user's mood
    selected_tracks = []
    for track in artist_tracks:
        if len(selected_tracks) == MAX_TRACKS: # number of tracks to queue
            break
        features = sp.audio_features([track['id']])[0]
        mood_criteria = MOOD_SETTINGS[user_mood]

        match = True
        for criteria, threshold in mood_criteria.items():
            if "min_" in criteria and features[criteria[4:]] < threshold:
                match = False
                break
            elif "max_" in criteria and features[criteria[4:]] > threshold:
                match = False
                break
        if match:
            print(f"{features}\n{mood_criteria}\n\n") # checking our work
            selected_tracks.append(track)

    track_names = [track['name'] for track in selected_tracks]
    track_list_str = "<br>".join(track_names)  # Using HTML line breaks for each track name
    print(track_list_str)
    track_uris = [track['uri'] for track in selected_tracks]
    sp.start_playback(device_id=device_id, uris=track_uris)

    return f"""
    **♫ Now Playing:** <span style="color: red; font-weight: bold; font-style: italic;">{artist_name}</span> ♫

    **Selected Tracks:**
    {track_list_str}
    """


### ### ### Recommendations ### ### ###
    

NUM_TRACKS = 10
def recommend_tracks(genre_name=None, artist_name=None, track_name=None, user_mood=None):
    """
    1. Retrieves user's preferences based on artist_name, track_name, genre_name, and user_mood.
    2. Uses these parameters to conduct personalized recommendations() search.
    """
    user_mood = get_user_mood(user_mood).lower() if user_mood else None
    print(user_mood)

    seed_artist, seed_track, seed_genre = None, None, None

    if genre_name:
        genre_name = get_genre_by_name(genre_name)
        seed_genre = [genre_name]
        print(seed_artist)

    if artist_name:
        results = sp.search(q='artist:' + artist_name, type='artist')
        seed_artist = [results['artists']['items'][0]['id']]

    if track_name:
        results = sp.search(q='track:' + track_name, type='track')
        seed_track = [results['tracks']['items'][0]['id']]
    
    recommendations = sp.recommendations(
                            seed_artists=seed_artist, 
                            seed_genres=seed_genre, 
                            seed_tracks=seed_track, 
                            limit=NUM_TRACKS,
                            country=None,
                            **MOOD_SETTINGS[user_mood] if user_mood else {})
    
    track_uris = [track['uri'] for track in recommendations['tracks']]
    return track_uris


def play_recommended_tracks(genre_name=None, artist_name=None, track_name=None, user_mood=None):
    track_uris = recommend_tracks(genre_name, artist_name, track_name, user_mood)
    track_list_str = create_track_list_str(track_uris) 
    sp.start_playback(device_id=device_id, uris=track_uris)

    return f"""
    **♫ Now Playing Recommendations Based On:** <span style="color: red; font-weight: bold; font-style: italic;">{', '.join(filter(None, [genre_name, artist_name, track_name, "Your Vibe"]))}</span> ♫

    **Selected Tracks:**
    {track_list_str}
    """


def create_playlist_from_recommendations(genre_name=None, artist_name=None, track_name=None, user_mood=None):
    user_id = sp.current_user()['id']
    user_name = sp.current_user()["display_name"]
    playlist_name = f"Apollo's Recommendations For {user_name}"
    playlist_description="Coming Soon."
    new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, collaborative=False, description=playlist_description)

    track_uris = recommend_tracks(genre_name, artist_name, track_name, user_mood)
    track_list_str = create_track_list_str(track_uris) 
    sp.user_playlist_add_tracks(user=user_id, playlist_id=new_playlist['id'], tracks=track_uris, position=None)
    playlist_url = f"https://open.spotify.com/playlist/{new_playlist['id']}"

    return f"""
    **♫ Created Recommendations Playlist Based On:** <span style="color: red; font-weight: bold; font-style: italic;">{', '.join(filter(None, [genre_name, artist_name, track_name, "Your Vibe"]))}</span> ♫

    **Selected Tracks:**
    {track_list_str}

    <a href='{playlist_url}'>Click here to listen to the playlist on Spotify!</a>
    """























