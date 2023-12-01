""" Loading lyrics and providing model context for the chatbot """
import os
from typing import Union
import pandas as pd
import numpy as np
import pinecone
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
import openai
import streamlit as st
from dotenv import load_dotenv


load_dotenv() # Load the .env file

# Read in the openai api key from the .env file
openai.api_key = os.getenv("OPENAI_KEY2")
# Read in the openai organization id from the .env file
openai.organization = os.getenv("OPENAI_ORG2")

embed = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key = openai.api_key,
    openai_organization = openai.organization,
    disallowed_special=()
)
# Initialize the session state
def init_session_variables():
    """Initialize session state variables"""
    session_vars = [
       'context', 'output', 'prompt'
    ]
    default_values = [
        [], None, ""
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize the session variables
init_session_variables()


def get_lyrics_vectorstore(index_name='lyrics', embeddings = embed):
    """Get the vectorstore from Pinecone"""
    # Initialize pinecone
    pinecone.init(api_key=os.getenv("PINECONE_KEY"), environment=os.getenv("PINECONE_ENV"))
    vectorstore = pinecone.Index(index_name, embeddings)
    
    return vectorstore

def get_music_vectorstore(index_name='musicgen'):
    """Get the vectorstore from Pinecone"""
    api_key = os.getenv("PINECONE_KEY2")
    environment = os.getenv("PINECONE_ENV2")
    pinecone.init(api_key = api_key, environment=environment)
    music_vectorstore = pinecone.Index(index_name)

    return music_vectorstore
    
def custom_csv_loader(file_path):
    """ Load the csv file and return a dataframe """
    return pd.read_csv(file_path, encoding='utf-8') 

def load_lyrics():
    """Load the lyrics from the csv file"""
    df = pd.read_csv('./lyrics/combs_lyrics.csv', encoding='utf-8')
    loader = DataFrameLoader(df, page_content_column="Song Title") # noqa

    data = loader.load() # Load the data

    return data


def get_artist_response(question:str, artist:str = "Dave Matthews"):
    """Get a response from the artist to the question"""
    pinecone.init(api_key = os.getenv("PINECONE_KEY"),
    environment=os.getenv("PINECONE_ENV"))  # Initialize pinecone
    vectorstore = get_lyrics_vectorstore(index_name='lyrics')
    context = get_context(vectorstore, question)
    context_dict = [{"Song Name": context.page_content,
                    "lyrics": context.metadata} for context in context]
    st.session_state.context = context_dict  # Cache the context
    messages = [
        {
            "role": "system", "content": f'''You are {artist}, the famous
            musician, helping a fellow musician out in a "co-writing" session
            where you are giving them advice based on your own style to help 
            them write songs. You have context {context_dict} pulled from your 
            song lyrics to help you relate to the user's question {question}. 
            Feel free to mention a specific song or lyrics of yours when guiding the users along.
            Your chat history so far is {st.session_state.chat_history}. 
            This will be a back and forth chat, so make sure to leave your responses open-ended.'''
        },
        {
            "role": "user", "content": f"Please answer my {question} about song writing."
        }
    ]
    
    models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo"]
    #models = ["ft:gpt-3.5-turbo-0613:david-thomas::7wEhz4EL"]
    full_response = ""
    
    for model in models:
        try:
            for response in openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=500,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1,
                stream=True
            ):
                full_response += response.choices[0].delta.get("content", "")
                if response.choices[0].delta.get("stop"):
                    break
            return full_response
        
        except Exception as e:
            print(e)
            continue

def get_context(vectorstore, question: str):
    """Get the context from the vectorstore"""
    context = vectorstore.similarity_search(
    query=question,
    k=3
    )

    return context

def get_inputs_from_llm(artist:str = "Dave Matthews"):
    """ We want the LLM to decide on the prompts for the 
    music generation model"""
    messages = [
        {
            "role": "system", "content": f"""You are {artist}, the famous
            musician, helping a fan out in a "co-writing" session
            where you are giving them advice based on your own style to help 
            them write songs.  The user would like you to help them create an
            audio sample based on your chat history {st.session_state.cowriter_messages}
            so far.  Based on the chat history, create a text prompt that
            would best identify to the music generation model what kind of song
            to create.  Remember you are {artist} when contemplating your answer.Think through
            how you can best represent the song you are helping the user create in your prompt."""
        },
        # Give some examples of prompts
        {
            "role": "system" , "content": '''Here are some examples of prompts for the music  
            generation model:
            1) "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms,
                perfect for the beach."
            2) "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and
                soaring strings, creating a cinematic atmosphere fit for a heroic battle."
            3) "reggaeton track, with a booming 808 kick, synth melodies layered with
            Latin percussion elements, uplifting and energizing"
            4) "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions"
            '''
        },
        {
            "role": "user", "content": """Please help me create a prompt for the
            music generation model based on our chat history."""
        },
    ]
    models = ["ft:gpt-3.5-turbo-0613:david-thomas::7wEhz4EL","gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613"] # Set list of models to iterate through
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages = messages,
                max_tokens=200,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                n=1
            )
            answer = response.choices[0].message.content
            st.session_state.prompt = answer
            return answer
        
        except TimeoutError as e:
            print(f"Timeout error: {e}")
            continue


def get_similar_audio_clips(audio_vector: Union[list, np.array]):
    """Get the most similar audio clips from the music vectorstore"""
    pinecone_key = os.getenv("PINECONE_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    pinecone.init(api_key = pinecone_key, environment=pinecone_env)
    index = pinecone.Index("musicgen")
    similar_audio = index.query(
        # Convert the audio vector to a list
        vector=audio_vector.tolist(),
        top_k=1,
        include_metadata=True,
        include_values=True
    )
    return similar_audio.to_dict()
