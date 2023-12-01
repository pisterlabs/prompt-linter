""" Functions for cowriter. """
from typing import Optional, List
import json
import os
from dotenv import load_dotenv
import streamlit as st
import lyricsgenius
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import openai
from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic, APIConnectionError
from pydantic import BaseModel, Field

# Load the environment variables
load_dotenv()

# Set the environment variables
GENIUS_API_KEY = os.getenv("GENIUS_TOKEN")
genius = lyricsgenius.Genius(GENIUS_API_KEY)

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

openai.organization = os.getenv("OPENAI_ORG2")
openai.api_key = os.getenv("OPENAI_KEY2")

# Pinecone keys
pinecone_api_key = os.getenv("PINECONE_KEY2")
pinecone_environment = os.getenv("PINECONE_ENV2")

pinecone.init(      
	api_key=pinecone_api_key,
    environment=pinecone_environment
)      
index = pinecone.Index('lyrics')

# Initiate the session state
if "lyrics_dict_list" not in st.session_state:
    st.session_state.lyrics_dict_list = None
if "upserts" not in st.session_state:
    st.session_state.upserts = None


# Create a class for the lyrics object
class SongClip(BaseModel):
    """ SongClip object. """
    lyrics: Optional[str] = Field(description="Lyrics of the song")
    title: str = Field(description="Title of the song")
    artist: str = Field(description="Artist of the song")
    album: Optional[str] = Field(description="Album of the song")
    source: str = Field(description="Source of the song")
    music: Optional[List[float]] = Field(
                description="Music of the song")
    description: Optional[str] = Field(
                description="Description of the song")

def format_claude_prompt(user_prompt):
    """ Format the prompt for the Claude model."""
     # The format for the prompt should be "Human:" \
    # followed by the user's prompt.  Then on a new line,
    # it should say "Assitant:"  followed by blank space.
    # The user prompt can include variables in the form 
    # {{variable_name}}.  As in {{YOUR TEXT HERE}}
    formatted_prompt=f"{HUMAN_PROMPT} {user_prompt} {AI_PROMPT}"
    return formatted_prompt

# Function to retrieve the lyrics from lyricsgenius
def get_lyrics(artist_name: str, n_songs: int = 1):
    """ Retrieve the songs by artist from lyricsgenius. """
    songs_dict = {} # Dictionary to store the songs
    # Search for the artist
    genius_songs = genius.search_artist(artist_name, max_songs=n_songs,\
                                    sort="popularity")
    # Iterate through the songs and create a SongClip object
    for song in genius_songs.to_dict()["songs"]:
        try:
            # Create a SongClip object
            clip = SongClip(lyrics=song["lyrics"],
            title=song["title"],
            artist=song["artist"],
            album=song["album"]["name"] if song["album"] else None,
            source=song["url"],
            description=song["description"]["plain"] if song["description"] else None
            )
            # Add the song to the dictionary
            songs_dict[song["title"]] = clip
        except TypeError as e:
            st.write(f"Could not retrieve lyrics for {song['title']}.  Error: {e}")
            continue    
    # Create a list of the songs
    songs_list = [song for song in songs_dict.values()]
    st.session_state.lyrics_dict_list = songs_list
        
    # Return the lyrics as a json object
    return songs_list

# Create a function to embed the lyrics and upsert
# them into our Pinecone index
def upsert_lyrics(songs_list: List[SongClip]):
    """ Upsert the lyrics into the Pinecone index. """
    embeddings = OpenAIEmbeddings()
        
    # Iterate through the songs and embed the lyrics
    for song in songs_list:
        # Embed the lyrics
        lyrics_emb = embeddings.embed_documents(song.lyrics)
        
        # Create the metadata
        text = song.lyrics
        metadata = {"title": song.title,
                    "artist": song.artist,
                    "album": song.album,
                    "source": song.source,
                    "description": song.description}
        values = lyrics_emb[0]
        song_id = song.title

        # Define the vectors
        vectors = [
            {
            "metadata": metadata,
            "values": values,
            "id": song_id,
            "text": text
            }
        ]

        # Upsert the lyrics
        index.upsert(vectors=vectors)

    # Return a success message and the index name statistics
    return f"Successfully upserted {len(songs_list)} songs into the index."
        
# Save the lyrics_dict_list to a json file
def save_lyrics(songs_list: List[SongClip]):
    """ Save the lyrics to a json file. """
    # Create a dictionary to store the lyrics
    lyrics_dict = {}
    # Iterate through the songs and add them to the dictionary
    for song in songs_list:
        lyrics_dict[song.title] = song.dict()
    # Save the lyrics to a json file
    with open("../tests/lyrics.json", "w") as f:
        json.dump(lyrics_dict, f)
    # Return a success message
    return f"Successfully saved {len(songs_list)} songs to lyrics.json"

# Load the lyrics from a json file
def load_lyrics():
    """ Load the lyrics from a json file. """
    # Load the lyrics from a json file
    with open("../tests/lyrics.json", "r") as f:
        lyrics_dict = json.load(f)
    # Create a list of the songs
    songs_list = [SongClip(**song) for song in lyrics_dict.values()]
    # Return the songs list
    return songs_list

# Define a function to perform search on Pinecone
def search(query: str, index_name=None, embeddings=OpenAIEmbeddings()):
    """ Perform search on Pinecone. """
    if not index_name:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        index_name = pinecone.Index('lyrics')
    # Embed the query
    query = embeddings.embed_query(query)
    # Perform search
    results = index_name.query(queries=[query], top_k=3, include_metadata=True)
    # Get song
    return results

def lyrics_chat_claude(query:str):
        """ Generate a response from our co-writer. """
        # Perform search
        k_results = search(query)
        # Convert the results to a dictionary
        results_dict_list = convert_results_to_dicts(k_results)
        claude_prompt = format_claude_prompt(f"""You are an famous artist and musician
        engaged in a co-writing session with another artist.
        They want to collaborate with you on a song they are working on.  Based on their
        query {query}, respond as you think a famous musician would.  Some of your song
        lyrics that may be relevant for context are {results_dict_list}.  Your
        conversation so far has been {st.session_state.messages}""")

        # Perform search
        k_results = search(query)
        # Convert the results to a dictionary
        results_dict_list = convert_results_to_dicts(k_results)
        i=0
        while i < 3:
            # Create the chat object
            try:
                response = anthropic.completions.create(
                    model="claude-2",
                    prompt=claude_prompt,
                    max_tokens_to_sample=300,
                    timeout=60,
                    temperature=1,
                    top_p=1,
                )
                # Get the response from Dave 
                dave_response = response.completion
                # Return the response
                return dave_response
            
            except APIConnectionError as e:
                print(e)
                if i==2:
                    return "I'm sorry, I'm having trouble connecting to the API.  Please try again later."
                else:
                    i+=1
                    continue

# Create an openai completion function
def lyrics_chat(query: str):
    """ Function to facilitate the chat between the user and the chatbot. """
    # Perform search
    k_results = search(query)
    # Convert the results to a dictionary
    results_dict_list = convert_results_to_dicts(k_results)
    # Construct the prompt
    messages = [
        {"role": "system", "content": f"""You are Dave Matthews engaged in a co-writing session
        with another artist.  They want to collaborate with you on a song they are working on.  Based on their
        query {query}, respond as you think Dave would.  Some lyrics that may be relevant for context
        are {results_dict_list}."""},
        {"role": "user", "content": f"{query}"}
    ]
    # Create a list of models to iterate through
    models = ["gpt-4-0613", "gpt-4", "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-16k, gpt-3.5-turbo-0613", "gpt-3.5-turbo"]
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                model=model,
                max_tokens=250,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0.5
            )
            artist_response = response.choices[0].message.content

            # Return the artist response
            return artist_response
        except TimeoutError as e:
            print(e)
            continue

def convert_results_to_dicts(search_res):
    """ Convert the Query Response Object to a list of dictionaries."""
    dicts_list = []
    results_dict = search_res.to_dict()
    for result in results_dict["results"][0]["matches"]:
        song = SongClip(**result["metadata"])
        dicts_list.append(song.dict())
        return dicts_list
