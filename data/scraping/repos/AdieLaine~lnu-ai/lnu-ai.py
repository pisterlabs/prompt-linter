# System-related
import os
import sys
import tempfile
import time

# File and IO operations
import json
import jsonlines
import librosa
import soundfile as sf

# Data processing and manipulation
import numpy as np
import pandas as pd
import random

# Data visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bokeh.plotting import figure
from scipy.signal import spectrogram
from streamlit_agraph import agraph, Node, Edge, Config

# Image processing
from PIL import Image

# Network requests
import requests

# Text-to-speech
from gtts import gTTS
from pydub import AudioSegment

# Type hints
from typing import Dict, List, Tuple, Optional

# Language model
import openai

# Web application framework
import streamlit as st
from streamlit_option_menu import option_menu

# Machine learning metrics and preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Environment variables
from dotenv import load_dotenv


CUSTOM_CSS = """
    <style>
        .big-font {
            font-size: 20px;
            font-weight: bold;
        }
        .red-text {
            color: crimson;
        }
        .green-text {
            color: #42f5ad;
        }
        .blue-text {
            color: #4287f5;
        }
        .selected-word {
            font-weight: bold;
            color: #f542f7;
        }
        .yellow-text {
            color: #FFD700;
        }
        .custom-option-menu select {
            font-weight: bold;
            color: #FF6347;
            background-color: #FFF;
        }
    </style>
"""


# Streamlit configurations
st.set_page_config(
    page_title="Lnu-AI - An Indigenous AI System",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/lnu-ai',
        'Report a bug': 'https://github.com/AdieLaine/lnu-ai/issues',
        'About': """
            # Lnu-AI
            Welcome to Lnu-AI! This application is dedicated to helping people learn and appreciate the Mi'kmaq language, an indigenous language of Eastern Canada and the United States. 

            ## About Mi'kmaq Language
            The Mi'kmaq language is a rich, polysynthetic language with a deep historical and cultural significance. It is, however, at risk of being lost as the number of fluent speakers decreases.

            ## The Lnu-AI Project
            Lnu-AI utilizes advanced AI technologies to provide a platform for learning, using, and preserving the Mi'kmaq language. It offers various features like chat functionality, storytelling, and deep linguistic analysis to facilitate language learning and appreciation.

            ## Your Contribution
            As an open-source project, we welcome contributions that help improve Lnu-AI and further its mission to preserve the Mi'kmaq language. Please visit our [GitHub](https://github.com/AdieLaine/lnu-ai) page for more information.
            
            Enjoy your journey with the Mi'kmaq language!
        """
    }
)


@st.cache_data
def load_all_word_details(file):
    """
    Load word details from a JSON file.

    Parameters:
    file (str): Path to the JSON file containing the word details.

    Returns:
    all_word_details (dict): Dictionary of word details loaded from the JSON file.
    """
    full_path = os.path.join("data", file)
    if os.path.isfile(full_path):
        with open(full_path, 'r') as f:
            all_word_details = json.load(f)
        return all_word_details
    else:
        return None
    
file = 'all_word_details.json'

all_word_details: dict = load_all_word_details(file)


@st.cache_data
def load_trained_data(file_paths):
    """
    Load trained data from multiple files. Handles both formats.

    Parameters:
    file_paths (list): List of paths to the trained data files.

    Returns:
    trained_data (list): List of trained data.
    """
    trained_data: list = []

    # Prioritize 'trained_data' in the list
    sorted_file_paths = sorted(file_paths, key=lambda path: "trained_data" != path)

    for file_path in sorted_file_paths:
        full_path = os.path.join("data", file_path)
        if os.path.isfile(full_path):
            with open(full_path, 'r') as f:
                trained_data.extend([json.loads(line) for line in f])

    if all('prompt' in data and 'completion' in data for data in trained_data):
        return trained_data
    else:
        return None

file_paths = ['trained_data.jsonl']

trained_data: list = load_trained_data(file_paths)


@st.cache_data
def load_word_details_embeddings(file):
    """
    Load word embeddings from a JSON file.

    Parameters:
    file (str): Path to the JSON file containing the word embeddings.

    Returns:
    word_details_embeddings (dict): Dictionary of word embeddings loaded from the JSON file.
    """
    full_path = os.path.join("data", file)
    if os.path.isfile(full_path):
        with open(full_path, 'r') as f:
            try:
                word_details_embeddings = json.load(f)
                return word_details_embeddings
            except json.JSONDecodeError:
                # Handle JSON decoding error
                return None
    else:
        # Handle file not found
        return None
    
file = 'word_details_embeddings.json'

word_details_embeddings: dict = load_word_details_embeddings(file)


@st.cache_data
def load_trained_data_embeddings(file_path):
    """
    Load trained data embeddings from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing the trained data embeddings.

    Returns:
    trained_data_embeddings (dict): Dictionary of trained data embeddings loaded from the JSON file.
    """
    full_path = os.path.join("data", file_path)
    if os.path.isfile(full_path):
        with open(full_path, 'r') as file:
            trained_data_embeddings = json.load(file)
        return trained_data_embeddings
    else:
        return None
    
file_path = 'trained_data_embeddings.json'

trained_data_embeddings: dict = load_trained_data_embeddings(file_path)


@st.cache_data
def load_theme_and_story(jsonl_file):
    """
    Load all themes from a JSONL file.

    Args:
        jsonl_file (str): Path to the JSONL file containing the themes.
        
    Returns:
        themes (list): A list containing all themes, or None if an error occurred.
    """
    try:
        full_path = os.path.join("data", jsonl_file)
        with jsonlines.open(full_path) as reader:
            themes = list(reader)  # Read all themes into a list
        return themes
    except (FileNotFoundError, jsonlines.jsonlines.InvalidLineError) as e:
        st.error(f"Error in loading themes: {str(e)}")
        return None

jsonl_file = "mikmaq_semantic.jsonl"

themes = load_theme_and_story(jsonl_file)


@st.cache_data
def load_word_data(file_path):
    """
    Load Lnu-AI Dictionary Word Data from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing the Lnu-AI Dictionary Word Data.

    Returns:
    word_data (dict): Dictionary of Lnu-AI Dictionary Word Data loaded from the JSON file.
    """
    data_path = os.path.join("data", file_path)
    try:
        with open(data_path, 'r') as file:
            word_data = json.load(file)  # Load and return the data
        return word_data
    except FileNotFoundError:  # Handle file not found error
        st.error(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:  # Handle JSON decode error
        st.error(f"Error decoding JSON file {file_path}.")
        return None


@st.cache_data
def load_embeddings(file_path):
    """
    Load embeddings from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing the embeddings.

    Returns:
    embeddings (np.array): Array of embeddings loaded from the JSON file.
    """
    data_path = os.path.join("data", file_path)
    try:
        with open(data_path, 'r') as file:
            embeddings = np.array(json.load(file))  # Load the embeddings
            embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Reshape the embeddings
            return embeddings
    except FileNotFoundError:  # Handle file not found error
        st.error(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:  # Handle JSON decode error
        st.error(f"Error decoding JSON file {file_path}.")
        return None
    except ValueError:  # Handle reshape error
        st.error(f"Error reshaping embeddings from file {file_path}.")
        return None


@st.cache_data
def find_most_similar_word(input_word, word_details_embeddings: dict):
    """
    Find the most similar word to the input word based on the cosine similarity of their embeddings.

    Parameters:
    input_word (str): The input word.
    word_details_embeddings (dict): Dictionary of word embeddings.

    Returns:
    (str): The most similar word to the input word, or the input word itself if no embedding is found for it.
    """
    input_embedding = word_details_embeddings.get(input_word)
    if input_embedding is not None:
        similarities = {word: 1 - cosine(input_embedding, embedding)
                        for word, embedding in word_details_embeddings.items() if word != input_word}
        most_similar_word = max(similarities, key=similarities.get)
        return most_similar_word
    return input_word  # If we don't have an embedding for the input word, just return the input word itself


@st.cache_data
def compute_cosine_similarity(vector1, vector2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    vector1, vector2 (list): Two vectors for which cosine similarity is to be computed.

    Returns:
    (float): Cosine similarity between the two vectors.
    """
    return cosine_similarity([vector1], [vector2])[0][0]


@st.cache_data
def calculate_cosine_similarity(embeddings, word_index1, word_index2):
    """
    Calculate the cosine similarity between two words based on their embeddings.

    Parameters:
    embeddings (np.array): Array of word embeddings.
    word_index1, word_index2 (int): Indices of the words in the embeddings array.

    Returns:
    (float): Cosine similarity between the two words.
    """
    try:
        vector1 = embeddings[word_index1].reshape(1, -1)  # Reshape the first vector
        vector2 = embeddings[word_index2].reshape(1, -1)  # Reshape the second vector
        return cosine_similarity(vector1, vector2)[0][0]  # Compute and return the cosine similarity
    except IndexError:  # Handle index out of range error
        st.error("Word index out of range for embeddings.")
        return None

@st.cache_data
def replace_unknown_words(user_message, word_details_embeddings: dict):
    """
    Replace unknown words in a message with the most similar known words.

    Parameters:
    user_message (str): The user's message.
    word_details_embeddings (dict): Dictionary of word embeddings.

    Returns:
    (str): The user's message with unknown words replaced with the most similar known words.
    """
    words = user_message.split()
    known_words = word_details_embeddings.keys()
    new_words = [word if word in known_words else find_most_similar_word(word, word_details_embeddings) for word in words]
    return ' '.join(new_words)


@st.cache_data
def clean_reply(reply):
    """
    Cleans the assistant's reply by removing trailing whitespaces and an extra period at the end, as well as unwanted "user" or "assistant" at the beginning.

    Parameters:
    reply (str): Reply from the assistant.

    Returns:
    str: Cleaned reply.
    """
    # Check if the reply starts with 'user:' or 'assistant:', and remove it if it does
    if reply.startswith('user:') or reply.startswith('assistant:'):
        reply = reply.split(':', 1)[1]

    # Split the reply into lines
    lines = reply.split('\n\n')

    # Remove trailing whitespace from the last line
    last_line = lines[-1].rstrip()

    # Check if the last line ends with '.' and remove it if it does
    if last_line.endswith("'""."):
        last_line = last_line[:-1]

    # Update the last line in the lines list
    lines[-1] = last_line

    # Join the lines back together
    cleaned_reply = '\n'.join(lines)

    return cleaned_reply.strip()  # Added strip() to remove leading/trailing whitespace


@st.cache_resource(show_spinner=False)
def load_env_variables_and_data():
    """
    Loads Lnu-AI Assistant environment variables and data.

    Returns:
    dict: A dictionary containing the loaded environment variables and data.
    """
    api_keys, tts_settings, local_data_files, models = load_env_variables()

    file_paths = ['trained_data.jsonl']
    trained_data = load_trained_data(file_paths)

    all_word_details = load_all_word_details(local_data_files.get("all_word_details"))
    trained_data_embeddings = load_trained_data_embeddings(local_data_files.get("trained_data_embeddings"))
    word_details_embeddings = load_word_details_embeddings(local_data_files.get("word_details_embeddings"))

    return {
        "api_keys": api_keys,
        "tts_settings": tts_settings,
        "local_data_files": local_data_files,
        "models": models,
        "completion_model": trained_data,
        "all_word_details": all_word_details,
        "trained_data": trained_data,
        "trained_data_embeddings": trained_data_embeddings,
        "word_details_embeddings": word_details_embeddings,
        "CUSTOM_CSS": CUSTOM_CSS
    }


@st.cache_resource(show_spinner=False)
def load_env_variables():
    """
    Load all the environment variables required for the Lnu-AI Assistant.

    Returns:
    Tuple: A tuple containing the loaded environment variables.
    """
    load_dotenv()

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "replicate": os.getenv("REPLICATE_API_TOKEN"),
    }

    tts_settings = {
        "tts_audio": os.getenv('TTS_AUDIO'),
        "eleven_labs": os.getenv('ELEVEN_LABS'),
        "speechki_audio": os.getenv('SPEECHKI_AUDIO'),
        "local_tts": os.getenv('LOCAL_TTS'),
    }

    local_data_files = {
        "trained_data": (os.getenv("TRAINED_DATA") + '.jsonl') if os.getenv("TRAINED_DATA") else None,
        "all_word_details": (os.getenv("ALL_WORD_DETAILS") + '.json') if os.getenv("ALL_WORD_DETAILS") else None,
        "trained_data_embeddings": (os.getenv("TRAINED_DATA_EMBEDDINGS") + '.json') if os.getenv("TRAINED_DATA_EMBEDDINGS") else None,
        "word_details_embeddings": (os.getenv("WORD_DETAILS_EMBEDDINGS") + '.json') if os.getenv("WORD_DETAILS_EMBEDDINGS") else None,
    }

    models = {
        "chat_model": os.getenv("CHAT_MODEL_SELECTION", default="gpt-4-0613"),
        "completion_model": os.getenv("COMPLETION_MODEL_SELECTION", default="text-davinci-003"),
        "fine_tuned_model_dictionary": os.getenv("FINE_TUNED_MODEL_DICTIONARY"),
        "fine_tuned_model_data": os.getenv("FINE_TUNED_MODEL_DATA"),
    }

    openai.api_key = api_keys["openai"]

    return api_keys, tts_settings, local_data_files, models


def generate_audio(text, tts_service):
    """
    Generate audio from text using a TTS service.

    Parameters:
    text (str): The input text.
    tts_service (str): The TTS service to use (e.g., "gtts", "st.audio").

    Returns:
    str: Path to the generated audio file.
    """
    tts_service = tts_service.lower()  # convert to lower case for case-insensitive comparison
    if tts_service == 'gtts':
        tts = gTTS(text=text, lang='en')  # Replace 'en' with the appropriate language if needed

        # Save the speech audio into a temporary mp3 file
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_mp3.name)  

        # Convert the temporary mp3 file to wav format
        audio = AudioSegment.from_mp3(temp_mp3.name)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format='wav')

        # Return the path to the generated wav file
        return temp_wav.name
    elif tts_service == 'st.audio':
        return text
    else:
        raise ValueError(f"Invalid Text-to-Speech service: {tts_service}")


def generate_openai_images(prompt, role="DALL-E", context="In the creative and vibrant style of Norval Morrisseau, using colorful Mi'kmaq themes"):
    """
    Generates an image using the OpenAI's DALL-E model.

    Args:
        prompt (str): The main role for the image generation.
        context (str, optional): Context to provide to the image generation.
                                 Defaults to "In the creative and vibrant style of Norval Morrisseau, using colorful Mi'kmaq themes".

    Returns:
        str: URL of the generated image if successful, else None.
    """
    try:
        full_prompt = f"{context} {prompt}"
        truncated_prompt = full_prompt[:300]
        prompt_settings = {
            "model": "image-alpha-001",
            "prompt": truncated_prompt,
        }
        response_settings = {
            "num_images": 1,
            "size": "1024x1024",
            "response_format": "url"
        }
        openai_key = os.getenv("OPENAI_API_KEY", "your_default_key")  # Use your default key here
        if not openai_key:
            st.info("Environment variable for OPENAI_API_KEY is not set, using default key: your_default_key")
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            },
            json={
                **prompt_settings,
                **response_settings
            }
        )
        response.raise_for_status()  # Raise an exception for any HTTP errors
        image_url = response.json()["data"][0]["url"]
        return image_url
    except (requests.RequestException, ValueError) as e:
        st.error(f"Error in generating images: {str(e)}")
        return None


@st.cache_data(experimental_allow_widgets=True)
def display_sound_of_words(user_selected_word, all_word_details: dict):
    """
    Process sound of words data for visualization.

    Args:
        user_selected_word (str): The selected word.
        all_word_details (dict): Dictionary of word details.

    Returns:
        None.
    """
    # If the word exists in the dictionary
    if user_selected_word in all_word_details:
        
        # Find wav files
        dir_path = os.path.join('media', 'audio', user_selected_word[0].lower(), user_selected_word)
        
        # If the directory exists
        if os.path.isdir(dir_path):
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            
            # If there are no WAV files
            if not wav_files:
                st.error(f"No WAV files found in the directory: {dir_path}")
                return

            selected_wav = st.sidebar.selectbox('Select the WAV file', wav_files)

            # If a wav file is selected
            if selected_wav:
                
                # Path to the selected wav file
                wav_file_path = os.path.join(dir_path, selected_wav)

                # Display the audio player in the sidebar
                st.sidebar.audio(wav_file_path)

                # Display selected visuals
                selected_visuals = st.sidebar.multiselect('Select visuals to display', 
                                                      options=['MFCC 3D Plot', 'Pitch Contour', 'Audio Waveform', 'Spectrogram'], 
                                                      default=['MFCC 3D Plot', 'Pitch Contour'])

                # Loading audio data, handling audio issues
                try:
                    y, sr = librosa.load(wav_file_path)
                except Exception as e:
                    st.error(f"Error loading the audio file: {wav_file_path}. Error: {e}")
                    return

                # Check if the audio data is valid
                if y.size == 0:
                    st.error(f"The audio file {wav_file_path} is empty or corrupted.")
                    return
                
                if 'MFCC 3D Plot' in selected_visuals:
                    # Compute MFCC features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                    # Create 3D plot for MFCC
                    fig_mfcc = go.Figure(data=go.Surface(z=mfccs, colorscale='Viridis', colorbar=dict(title='MFCC')))
                    fig_mfcc.update_layout(
                        title='MFCC (3D view)', 
                        scene=dict(
                            xaxis_title='Time',
                            yaxis_title='MFCC Coefficients',
                            zaxis_title='MFCC Value'
                        )
                    )
                    # Display the plot
                    st.plotly_chart(fig_mfcc)
                    st.info("The 3D plot above is a representation of MFCC (Mel-frequency cepstral coefficients). It's a type of feature used in sound processing. The axes represent Time, MFCC Coefficients, and MFCC Value.")

                if 'Pitch Contour' in selected_visuals:
                    # Compute the spectral centroid and pitch
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

                    # Prepare data for 3D plot
                    x = np.array([i for i in range(pitches.shape[1]) for _ in range(pitches.shape[0])])
                    y = np.array([i for _ in range(pitches.shape[1]) for i in range(pitches.shape[0])])
                    z = pitches.flatten()
                    pitch_mag = magnitudes.flatten()

                    # Create color and size based on pitch magnitude
                    color = pitch_mag / np.max(pitch_mag)
                    size = pitch_mag * 50 / np.max(pitch_mag)

                    # Create 3D plot for pitch contour
                    fig_pitch = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=size, color=color, colorscale='Viridis', opacity=0.8))])
                    fig_pitch.update_layout(scene=dict(xaxis_title='Time (s)', yaxis_title='Frequency Bin', zaxis_title='Pitch (Hz)'), title=f'Pitch Contour: {user_selected_word}')
                    st.plotly_chart(fig_pitch)
                    st.info("The 3D plot above is a representation of the pitch contour. It's a way of visualizing the variation in pitch with respect to time.")

                if 'Audio Waveform' in selected_visuals:
                    # Display audio waveform
                    st.subheader("Audio Waveform")
                    waveform_fig = view_audio_waveform(wav_file_path)
                    st.pyplot(waveform_fig)
                    st.info("The Audio Waveform is a graphical representation of the amplitude of the sound wave against time.")

                if 'Spectrogram' in selected_visuals:
                    # Display spectrogram
                    st.subheader("Spectrogram")
                    spectrogram_fig = generate_audio_spectrogram(wav_file_path)
                    st.pyplot(spectrogram_fig)
                    st.info("The Spectrogram represents how the frequencies of the audio signal are distributed with respect to time.")
                
        else:
            st.info(f"No directory found for the word {user_selected_word} at the expected location: {dir_path}")

            # filter the words that begin with 'q'
            q_words = [(word, all_word_details[word]['meanings']) for word in all_word_details.keys() if word[0].lower() == 'q']

            # create a markdown string with words and their meanings as a bullet point list
            # strip the square brackets and single quotes from meanings
            q_words_info = '\n'.join([f'- <span style="color:red;">{word}</span>: <span style="color:white;">{str(meanings)[2:-2]}</span>' for word, meanings in q_words])

            st.markdown(f"We have limited the words for to demonstrate the function. These words are available to use:\n{q_words_info}", unsafe_allow_html=True)



            
    else:
        st.error(f"The word {user_selected_word} not found in the database.")


def view_audio_waveform(input_audio):
    """
    Generate a waveform visualization of the audio.

    Parameters:
        input_audio (str): Path to the input audio file.

    Returns:
        fig (matplotlib.figure.Figure): The generated waveform plot figure.
    """
    waveform, sample_rate = sf.read(input_audio)
    time = np.arange(0, len(waveform)) / sample_rate

    fig = plt.figure(figsize=(10, 4))
    plt.plot(time, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')

    return fig


def generate_audio_spectrogram(input_audio):
    """
    Generate a spectrogram visualization of the audio.

    Parameters:
        input_audio (str): Path to the input audio file.

    Returns:
        fig (matplotlib.figure.Figure): The generated spectrogram plot figure.
    """
    waveform, sample_rate = sf.read(input_audio)
    _, _, Sxx = spectrogram(waveform, fs=sample_rate)

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(Sxx + 1e-10), aspect='auto', cmap='inferno', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')

    return fig


def generate_audio_visualization(input_audio, audio_info):
    """
    Generate a 3D visual representation of the audio.

    Parameters:
        input_audio (str): Path to the input audio file.
        audio_info (dict): Dictionary containing audio information (file_name, sample_rate, duration).

    Returns:
        fig (plotly.graph_objects.Figure): The generated 3D audio visualization figure.
    """
    waveform, sample_rate = sf.read(input_audio)
    _, _, Sxx = spectrogram(waveform, fs=sample_rate)

    fig = go.Figure(data=[go.Surface(z=np.log10(Sxx + 1e-10), colorscale='inferno')])
    fig.update_layout(scene=dict(
        xaxis_title='Time',
        yaxis_title='Frequency',
        zaxis_title='Intensity',
        aspectratio=dict(x=1, y=1, z=0.3),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        zaxis=dict(showgrid=False, showticklabels=False),
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=0.8),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0)
        )
    ))

    annotation_text = f"Audio File: {audio_info['file_name']}\n" \
                      f"Sample Rate: {audio_info['sample_rate']} Hz\n" \
                      f"Duration: {audio_info['duration']} seconds"
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=annotation_text,
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    return fig


@st.cache_data(show_spinner=True)
def generate_word_translations_visualization(embeddings, word_list, all_word_details, selected_word, num_nodes, edge_option, previous_node_ids):
    """
    Generates a visualization of word translations based on vector embeddings.
    
    Parameters:
        embeddings (numpy.ndarray): Array of word embeddings.
        word_list (list): List of words corresponding to the embeddings.
        all_word_details (dict): Dictionary containing additional details for each word.
        selected_word (str): The selected word for visualization.
        num_nodes (int): The number of closest words to include in the visualization.
        edge_option (list): List of edge options to include in the visualization.
        previous_node_ids (set): Set of previously added node IDs.

    Returns:
        tuple: A tuple containing the nodes, edges, config, and updated previous_node_ids.
    """

    # Fetch index of selected word
    word_index = word_list.index(selected_word)

    # Initialize nodes and edges
    nodes = []
    edges = []

    # Create the main node for the selected word
    main_word_node = Node(id=selected_word, label=selected_word, shape="box", font={"size": 20}, color="crimson")
    
    # Add the main node to the nodes list if not already added
    if main_word_node.id not in previous_node_ids:
        nodes.append(main_word_node)
        previous_node_ids.add(main_word_node.id)

    # Calculate cosine similarities and get indices of closest words
    closest_word_indices = np.argsort([calculate_cosine_similarity(embeddings, word_index, i) for i in range(len(word_list))])[::-1]


    # Define node colors for different edge options
    node_colors = {"translation": "cyan", "part_of_speech": "limegreen", "pronunciation": "red", "meanings": "gold"}

    # Create nodes and edges for closest words
    for i in range(1, num_nodes + 1):
        closest_word = word_list[closest_word_indices[i]]
        closest_word_node = Node(id=closest_word, label=closest_word, shape="box", font={"size": 12}, color="lavender")

        # Add closest word node to the nodes list if not already added
        if closest_word_node.id not in previous_node_ids:
            nodes.append(closest_word_node)
            previous_node_ids.add(closest_word_node.id)

        # Create an edge between the selected word and the closest word
        translation_edge = Edge(source=selected_word, target=closest_word)
        edges.append(translation_edge)

        # Create additional nodes and edges for selected edge options
        for key in edge_option:
            if key in all_word_details[closest_word]:
                value = all_word_details[closest_word][key]
                formatted_key = key.replace("_", " ").title()
                if formatted_key == 'Part Of Speech':
                    formatted_key = 'Part of Speech'
                key_node = Node(id=f"{closest_word}-{key}", label=f"{formatted_key}: {value}", shape="box", font={"size": 10}, color=node_colors[key])

                # Add key node to the nodes list if not already added
                if key_node.id not in previous_node_ids:
                    nodes.append(key_node)
                    previous_node_ids.add(key_node.id)

                # Create an edge between the closest word and the key node
                translation_edge = Edge(source=closest_word, target=f"{closest_word}-{key}")
                edges.append(translation_edge)

    # Create a configuration object for the graph
    config = Config(width=1024, height=1024, directed=True, hierarchical=True, center_nodes=True)

    return nodes, edges, config, previous_node_ids


def explore_words_multiselect(key, label, options, default, format_func):
    """
    Creates a multiselect dropdown in the sidebar with session state.

    Parameters:
        key (str): Unique identifier for the session state variable.
        label (str): Label to display above the dropdown menu.
        options (list): List of options for the dropdown menu.
        default (list): Default selection for the dropdown menu.
        format_func (function): Function to format the display of the options.
    """

    # Initialize session state with default values if the key doesn't exist
    if key not in st.session_state:
        st.session_state[key] = default

    # Create a multiselect dropdown menu
    selection = st.sidebar.multiselect(
        label=label,
        options=options,
        default=st.session_state[key],
        format_func=format_func,
        help=":bulb: Select one or more options for the edge."
    )

    # If "All Nodes" is selected, include all edge options in the state
    if "all" in selection:
        selection = options

    # Update the session state when the selection changes
    if st.session_state[key] != selection:
        st.session_state[key] = selection


@st.cache_data
def explore_word_details(selected_word, all_word_details):
    """
    Displays details of a selected word in the sidebar. If the word does not exist, an error message is displayed.

    Parameters:
        selected_word (str): Word selected by the user.
        all_word_details (dict): Dictionary of all word details.
    """

    # If the selected word does not exist, display an error message and return
    if selected_word not in all_word_details:
        st.error(f"The word {selected_word} does not exist in the dictionary.")
        return

    # Get details for the selected word
    word_detail = all_word_details[selected_word]
    word = word_detail.get("word", "")
    pronunciation = word_detail.get("pronunciation", "")
    part_of_speech = word_detail.get("part_of_speech", "")
    translation = word_detail.get("translation", "")
    meanings = word_detail.get("meanings", ["N/A"])
    example = word_detail.get("example", ["N/A"])
    alternate_forms = word_detail.get("alternate_forms", ["N/A"])


    # Only display details if they exist
    if word:
        st.markdown(f"<div style='color:CornflowerBlue;'>Word: </div>{word}", unsafe_allow_html=True)
    if pronunciation:
        st.markdown(f"<div style='color:CornflowerBlue;'>Pronunciation: </div>{pronunciation}", unsafe_allow_html=True)
    if part_of_speech:
        st.markdown(f"<div style='color:CornflowerBlue;'>Part of Speech: </div>{part_of_speech}", unsafe_allow_html=True)
    if translation:
        st.markdown(f"<div style='color:CornflowerBlue;'>Translation: </div>{translation}", unsafe_allow_html=True)

    st.markdown("<div style='color:CornflowerBlue;'>Meanings:</div>", unsafe_allow_html=True)
    for meanings in meanings:
        st.markdown(f"- {meanings}", unsafe_allow_html=True)

    st.markdown("<div style='color:CornflowerBlue;'>Example of word used in sentence:</div>", unsafe_allow_html=True)
    for example_line in example:
        example_parts = example_line.split('\n')
        if len(example_parts) >= 3:
            st.markdown(
                f"{example_parts[0]} <br> {example_parts[1]} <br> <span style='color:white;'>{example_parts[2]}</span>",
                unsafe_allow_html=True
            )


def explore_words():
    """
    Loads the word data and embeddings, creates an interactive search feature, 
    and displays visualizations based on user selection.
    """

    # Set up custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Define file path for "bnr2.png"
    image_path = os.path.join(os.getcwd(), "images", "bnr2.png")

    # Load and display the image in the sidebar
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True, clamp=True, channels="RGB", output_format="png")

    # Instruction text
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <div class="big-font -text center-text" style="color: crimson; font-size: 24px;">Examine Words</div>
        </div>
        <ul>
            <li>Type an English word into the 'Search for a word' field.</li>
            <li>Explore other Mi'kmaq words with similar meaning.</li>
            <li>Select a Method to Explore</li>
        </ul>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.info("The Vector Matrix of all Mi'kmaq Words\n\n- Explore New Word Connections\n- See and Hear the Essence of Mi'kmaq Orthography\n- Dive into the Frequency of Mi'kmaq Words\n- What Makes a Word a Word?")
    
    # Display the image in the sidebar
    search_word = st.sidebar.text_input("Type an English word", value="")

    # Instruction text
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="big-font">
                <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Examine</span>
                <span style="color:white;">:</span>
                <span style="font-size:24px; color:crimson;">Panuijgatg</span>
            </div>
            <ul style="text-align: left;">
                <li>Each Mi'kmaq word carries a linguistic tapestry of cultural significance and historical depth.</li>
                <li>Mi'kmaq words harmoniously blend sound, syntax, and semantics to convey profound concepts.</li>
                <li>The intricacy of Mi'kmaq words reveals the linguistic ingenuity of the Mi'kmaq people.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Custom CSS for the image and title
    st.markdown("""
        <style>
        .explore-words-title {
            font-size: 24px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Define file paths
    file_path = 'all_word_details.json'

    embeddings_path = 'embedded_word_data.json'

    # Load word data and embeddings
    all_word_details = load_word_data(file_path)

    embeddings = load_embeddings(embeddings_path)

    # If loading fails, terminate the function
    if all_word_details is None or embeddings is None:
        return

    # Generate word list and transform data to DataFrame
    word_list = list(all_word_details.keys())
    df = pd.DataFrame([(key, value["translation"], value["meanings"], value["example"]) for key, value in all_word_details.items()],
                        columns=["word", "translation", "meanings", "example"])

    # Sort the DataFrame by 'word' column in ascending order
    df = df.sort_values(by=['word'])

    # Resetting the index
    df = df.reset_index(drop=True)

    similar_words = None
    selected_word = ""
    user_selected_word = ""

    # Check if a search_word has been entered
    if search_word:  
        search_word_lower = search_word.lower()

        # Find the exact match in the "translation" field where the searched word is the first word
        exact_match_translation = df[df['translation'].apply(lambda translation: translation.lower().split()[0] == search_word_lower)]['word'].values

        if len(exact_match_translation) > 0:
            selected_word = exact_match_translation[0]
        else:
            # Find the word in the "meanings" field where the searched word is present
            similar_words_meanings = df[df['meanings'].apply(lambda meanings: any(search_word_lower in meanings.lower().split() for meanings in meanings))]['word'].values
            if len(similar_words_meanings) > 0:
                selected_word = similar_words_meanings[0]

        if not selected_word:
            st.sidebar.write("No similar word found.")
        
        if selected_word:
            # Get index of the selected word in dataframe
            selected_word_index = df.index[df['word'] == selected_word].tolist()[0]

            # Get next 19 words after the selected word
            next_words = df.iloc[selected_word_index+1 : selected_word_index+20]['word'].values

            # Combine the selected word with next 19 words
            combined_words = np.concatenate(([selected_word], next_words))

            user_selected_word = st.sidebar.selectbox("Similar words:", combined_words)


    visual_option = st.sidebar.selectbox("Select a method to explore", ("Word Visualization", "Sound of Words"))

    # Check if 'Sound of Words' is not selected
    if visual_option != "Sound of Words":
        
        # Edge option dropdown, number of nodes slider
        explore_words_multiselect(
            key="edge_option",
            label=":gear: Edge Option",  # Adding an emoji to the label
            options=["all", "translation", "part_of_speech", "pronunciation", "meanings"],
            default="all",
            format_func=lambda x: "All Nodes" if x == "all" else {
                "translation": "Translation",
                "part_of_speech": "Part of Speech",
                "pronunciation": "Pronunciation",
                "meanings": "Meanings",
            }.get(x, "Unknown")
        )

        num_nodes = st.sidebar.slider("Number of nodes", 5, 200, 10)

        # Check if 'All Nodes' is selected
        if "all" in st.session_state["edge_option"]:
            st.session_state["edge_option"] = ["translation", "part_of_speech", "pronunciation", "meanings"]

    # Initialize set to keep track of added node IDs
    previous_node_ids = set()

    # Display word details in sidebar
    if user_selected_word:
        with st.sidebar:
            explore_word_details(user_selected_word, all_word_details)

    # Auto-render the first visual in the main area
    if visual_option == "Word Visualization":
        if user_selected_word:
            # Call the generate_word_translations_visualization function
            nodes, edges, config, previous_node_ids = generate_word_translations_visualization(embeddings, word_list, all_word_details, user_selected_word, num_nodes, st.session_state["edge_option"], previous_node_ids)
            agraph(nodes=nodes, edges=edges, config=config)
            
    elif visual_option == "Sound of Words":
            # Call the display_sound_of_words function
            display_sound_of_words(user_selected_word, all_word_details)


@st.cache_resource(show_spinner=False)
def enhance_with_gpt(prompt, final_reply, models):
    """
    Enhances the reply with GPT model by sending the conversation for completion.

    Args:
        prompt (str): User's message.
        final_reply (str): Assistant's reply.
        models (dict): Dictionary of trained models.

    Returns:
        str: Enhanced reply.
    """
    model_name = models.get("chat_model", "gpt-4-0613")
    try:
        gpt_messages = [
            {"role": "system", "content": "You are Lnu-AI, an AI developed to promote and preserve the Mi'kmaq language and culture."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_reply}
        ]
    
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=gpt_messages,
            max_tokens=1200,  
            temperature=0.5, 
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )

        # Extract reply from response
        if 'choices' in response and len(response['choices']) > 0:
            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                gpt_reply = response['choices'][0]['message']['content']
                gpt_reply = clean_reply(gpt_reply) 
                return gpt_reply

    except Exception as e:
        print(f"Error with {model_name}: {e}")

    return final_reply


@st.cache_data(show_spinner=True)
def lnu_ai_chat(prompt: str, trained_data: list, models: dict, word_details_embeddings: dict) -> str:
    """
    Generates a response for a given user prompt using trained data or a language model. 
    Ensures that unknown words in the user's message are replaced with the most similar known words.

    Args:
        prompt (str): The message inputted by the user.
        trained_data (list): A list of dictionaries containing prompts and their corresponding completions from training data.
        models (dict): A dictionary of trained models for the chat bot.
        word_details_embeddings (dict): A dictionary of word embeddings for known words.

    Returns:
        str: The generated response from the chat bot.
    """
    # Replace unknown words in the user's prompt
    prompt = replace_unknown_words(prompt.lower(), word_details_embeddings)

    # Check for matches in trained data
    matched_prompt_data = [data for data in trained_data if prompt == data['prompt'].lower()]
    
    if matched_prompt_data:

        fine_tuned_replies = [data['completion'] for data in matched_prompt_data]
        reply = '\n\n'.join(fine_tuned_replies)
        reply = clean_reply(reply)
        
        # Enhance the reply using GPT
        reply = enhance_with_gpt(prompt, reply, models)

        # Add the new reply to the trained_data
        if reply not in fine_tuned_replies:
            new_prompt_data = {'prompt': prompt, 'completion': reply}
            trained_data.append(new_prompt_data)

        return reply

    # If no match found in trained data, generate completion with the trained model
    model_to_use = models["fine_tuned_model_data"] if "fine_tuned_model_data" in models else models.get("completion_model", "curie:ft-personal-2023-05-16-05-11-43")
    
    response = openai.Completion.create(
        model=model_to_use,
        prompt=prompt,
        max_tokens=1000,
        temperature=0.4,
    )
    
    if 'choices' in response and len(response['choices']) > 0:
        if 'text' in response['choices'][0]:
            final_reply = response['choices'][0]['text']
            final_reply = clean_reply(final_reply)
            
            # Enhance the reply using GPT
            final_reply = enhance_with_gpt(prompt, final_reply, models)

            # Add this new prompt-reply pair to the trained_data
            new_prompt_data = {'prompt': prompt, 'completion': final_reply}
            trained_data.append(new_prompt_data)

            return final_reply
    
    # Default response if no response is generated
    return "Sorry, I don't understand. Please try again."


def display_detail(label, value):
    """
    Helper function to display a single detail of a word.

    Args:
        label (str): Label for the detail.
        value (str): Value of the detail.
    """
    if value is None or value == "N/A":
        value = "Not available"
    st.markdown(f"<div style='color:CornflowerBlue;'>{label}: </div>{value}", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def display_word_details_chatbot(selected_word, all_word_detail: dict):
    """
    Displays details of a selected word in the chatbot interface.

    Parameters:
        selected_word (str): Selected word.
        all_word_detail (dict): Details of all words.
    """
    if selected_word not in all_word_detail:
        st.error(f"The word {selected_word} does not exist in the dictionary.")
        return

    word_detail = all_word_detail[selected_word]
    word = word_detail.get("word", "N/A")
    pronunciation = word_detail.get("pronunciation", "N/A")
    part_of_speech = word_detail.get("part_of_speech", "N/A")
    translation = word_detail.get("translation", "N/A")
    meanings = word_detail.get("meanings", ["N/A"])
    example = word_detail.get("example", ["N/A"])
    alternate_forms = word_detail.get("alternate_forms", ["N/A"])

    # Display the word details using the helper function
    display_detail('Word', word)
    display_detail('Pronunciation', pronunciation)
    display_detail('Part of Speech', part_of_speech)
    display_detail('Translation', translation)

    # Display meanings
    st.markdown("<div style='color:CornflowerBlue;'>Meanings:</div>", unsafe_allow_html=True)
    for meanings in meanings:
        st.markdown(f"- {meanings}", unsafe_allow_html=True)
    st.sidebar.markdown("---")


def chatbot_application(models, trained_data: dict, local_data_files: dict, tts_settings: dict):
    """
    Main chatbot application function. It loads previous session states or initializes new ones,
    sets up the sidebar and the main chat area, and handles user input and chatbot responses.

    Parameters:
        models (dict): Dictionary of trained models.
        trained_data (dict): Dictionary of trained data.
        local_data_files (dict): Dictionary of paths to local data files.
        tts_settings (dict): Text-to-speech settings.
    """
    # Custom CSS for the image and title
    st.markdown("""
        <style>
        .explore-words-title {
            font-size: 24px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Define file paths
    image_path = os.path.join(os.getcwd(), "images", "bnr4.png")
    
    # Load and display images in the sidebar
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True, clamp=True, channels="RGB", output_format="png")

    # Instruction text
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <div class="big-font -text center-text" style="color: crimson; font-size: 24px;">Lnu-AI Chat</div>
        </div>
        <ul>
            <li>I am trained on all Mi'kmaq words.</li>
            <li>Ask me about Mi'kmaq culture or history.</li>
            <li>I will do my best to answer any question you have.</li>
        </ul>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Ask me about:\n\n- Glooscap\n- The Mi'kmaq Creation Story\n- Mi'kmaq Linguistics\n- How to make a Canoe\n- Craft a Sweetgrass Basket\n- Mi'kmaq History")

    # Sidebar: search field
    search_word = st.sidebar.text_input("Search for a word", value="")

    # Load previous session states or initialize new ones
    st.session_state.setdefault('generated', [])
    st.session_state.setdefault('past', [])

    # Load the trained data
    try:
        trained_data_file = os.path.join("data", "trained_data.jsonl")
        with open(trained_data_file, "r") as f:
            trained_data = [json.loads(line) for line in f]

        word_details_data = load_all_word_details(local_data_files["all_word_details"])
        trained_data_embeddings = load_trained_data_embeddings(local_data_files["trained_data_embeddings"])
        word_details_embeddings = load_word_details_embeddings(local_data_files["word_details_embeddings"])
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Define file paths
    file_path = 'all_word_details.json'
    embeddings_path = 'embedded_word_data.json'
    trained_data = 'trained_data.jsonl'
    
    # Load word data and embeddings
    all_word_details = load_word_data(file_path)
    embeddings = load_embeddings(embeddings_path)
    trained_data = load_trained_data(file_paths)

    # If loading fails, terminate the function
    if all_word_details is None or embeddings is None:
        return

    # Generate word list and transform data to DataFrame
    word_list = list(all_word_details.keys())
    df = pd.DataFrame([(key, value["translation"], value["meanings"], value["example"]) for key, value in all_word_details.items()],
                        columns=["word", "translation", "meanings", "example"])

    # Sort the DataFrame by 'word' column in ascending order
    df = df.sort_values(by=['word'])

    # Resetting the index
    df = df.reset_index(drop=True)

    similar_words = None
    selected_word = ""
    user_selected_word = ""

    # Check if a search_word has been entered
    if search_word:  
        search_word_lower = search_word.lower()

        # Find the exact match in the "translation" field where the searched word is the first word
        exact_match_translation = df[df['translation'].apply(lambda translation: translation.lower().split()[0] == search_word_lower)]['word'].values

        if len(exact_match_translation) > 0:
            selected_word = exact_match_translation[0]
        else:
            # Find the word in the "meanings" field where the searched word is present
            similar_words_meanings = df[df['meanings'].apply(lambda meanings: any(search_word_lower in meanings.lower().split() for meanings in meanings))]['word'].values
            if len(similar_words_meanings) > 0:
                selected_word = similar_words_meanings[0]

        if not selected_word:
            st.sidebar.write("No similar word found.")
        
        if selected_word:
            # Get index of the selected word in dataframe
            selected_word_index = df.index[df['word'] == selected_word].tolist()[0]

            # Get next 19 words after the selected word
            next_words = df.iloc[selected_word_index+1 : selected_word_index+20]['word'].values

            # Combine the selected word with next 19 words
            combined_words = np.concatenate(([selected_word], next_words))

            user_selected_word = st.sidebar.selectbox("Similar words:", combined_words)

    # Display word details in sidebar
    if user_selected_word:
        with st.sidebar:
            display_word_details_chatbot(user_selected_word, all_word_details)

    # Instruction text
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="big-font">
                <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Talk Together</span>
                <span style="color:white;">:</span>
                <span style="font-size:24px; color:crimson;">Mawagnutmajig</span>
            </div>
            <ul style="text-align: left;">
                <li>I’m Lnu-AI, your Mi'kmaq AI Assistant.</li>
                <li>I am still learning and won’t always get it right, but our conversations will help me improve.</li>
                <li>It serves as a means to preserve and transmit knowledge, values, beliefs, and history from one generation to the next.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    with st.container():
        # Define color variables
        question_color = "CornflowerBlue"
        lnu_ai_color = "crimson"
        background_color = "#262626"

        # Chat form for user input
        with st.form(key='chat_form', clear_on_submit=True):
            user_message = st.text_area("Chat with Lnu-AI", value="", height=150, max_chars=500, key="chat_input_chat")
            submit_button = st.form_submit_button("Send a message")

        # Process user input and display chatbot response
        if submit_button and user_message:
            user_message = user_message.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            st.session_state['past'].append(user_message)
            with st.spinner("Referring to my Mi'kmaq Corpus ..."):
                chat_response = lnu_ai_chat(user_message, trained_data, models, all_word_details)
                st.session_state['generated'].append(chat_response)
                
                # Generate audio response and play it
                tts_service = tts_settings.get('tts_audio', 'gtts')
                audio_file = generate_audio(chat_response, tts_service)
                audio_response = open(audio_file, "rb")
                st.audio(audio_response.read(), format="audio/wav")
                os.remove(audio_file)

        # Display the chat history
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            chat_response = st.session_state['generated'][i]
            user_message = st.session_state['past'][i]
            if chat_response:  # Only display non-empty messages
                st.markdown(
                    f'<div style="background-color: {background_color}; padding: 20px; border-radius: 20px;">'
                    f'<b style="color: {question_color};">Question:</b> {user_message}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background-color: {background_color}; padding: 20px; border-radius: 20px;">'
                    f'<b style="color: {lnu_ai_color};">Lnu-AI:</b> {chat_response}</div>',
                    unsafe_allow_html=True
                )


def display_mikmaq_resources() -> None:
    """
    Display Mi'kmaq language resources in a Streamlit application.
    """
    # Set up the sidebar
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Define file paths
    image_path = os.path.join(os.getcwd(), "images", "bnr3.png")

    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width="always", clamp=True, channels="RGB", output_format="png")

    # Instruction text
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <div class="big-font -text center-text" style="color: crimson; font-size: 24px;">Language Preservation</div>
        </div>
        <ul></ul>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    page_selection = st.sidebar.radio("Go to", ["Home", "Helpful Links", "The Lnu-AI Project"])

    st.sidebar.info("The Lnu-AI Project is open source. [Lnu-AI Repository](https://github.com/AdieLaine/lnu-ai)")
    st.sidebar.info("Connect with the Developer on [Twitter](https://twitter.com/justmadielaine).")
    if page_selection == "Home":
        
        # Instruction text
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                <div class="big-font">
                    <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Preserve</span>
                    <span style="color:white;">:</span>
                    <span style="font-size:24px; color:crimson;">Gweso'tg</span>
                </div>
                <ul style="text-align: left;">
                    <li>Mi'kmaq language resources provide invaluable tools for learning, studying, and immersing oneself in the language.</li>
                    <li>Collaboration among linguists, educators, and community members strengthens the development and availability of Mi'kmaq language resources.</li>
                    <li>The commitment to Mi'kmaq language preservation and the availability of resources empower individuals to engage with and contribute to the revitalization of their language.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

    elif page_selection == "The Lnu-AI Project":
        image_path = os.path.join(os.getcwd(), "images", "LnuSideFlag.png")

        # Load and display the image
        image = Image.open(image_path)
        st.image(image, use_column_width="always", clamp=True, channels="RGB", output_format="png")
        st.markdown("""
            ## The :red[Lnu-AI] Project

            The :red[Lnu-AI] Project is a groundbreaking initiative that creatively combines the power of Artificial Intelligence systems with a profound respect for cultural heritage to revolutionize the preservation and revitalization of the Mi'kmaq language. With its roots firmly anchored in the Mi'kmaq community, :red[Lnu-AI] leverages cutting-edge technologies, such as machine learning and natural language processing, to ensure the survival and advancement of this culturally significant language.

            In an era where countless indigenous languages are on the brink of extinction, preserving them becomes an imperative mission. The Mi'kmaq language, echoing the voice of the Mi'kmaq people across Mi'kma'ki, is a living testament to a rich history and culture. The :red[Lnu-AI] Project endeavors to construct a dynamic platform that safeguards this linguistic legacy, fostering an environment where the Mi'kmaq language can flourish and be appreciated by both the Mi'kmaq community and the wider world.

            At its core, :red[Lnu-AI] aims to offer meticulously accurate translations, definitions, and contextual applications of Mi'kmaq words and phrases. :red[Lnu-AI] uses sophisticated machine learning algorithms to train on an extensive dataset drawn from various Mi'kmaq language resources. This method guarantees :red[Lnu-AI]'s proficiency in comprehending the Mi'kmaq language, enabling it to generate specific and culturally relevant content. Whether you're an enthusiastic language learner, a cultural explorer, or intrigued by the Mi'kmaq language, :red[Lnu-AI] can provide accurate and detailed responses.

            The scope of the :red[Lnu-AI] Project extends beyond just language preservation. Recognizing the connection between language and culture, :red[Lnu-AI] transcends typical translation services as a comprehensive digital archive of Mi'kmaq culture, traditions, history, and customs. This rich repository fuels a deeper understanding and appreciation of the Mi'kmaq lifestyle, with :red[Lnu-AI] capable of enlightening users about traditional ceremonies, folklore, art, music, and much more. This fusion of past language meanings and present-day AI solutions, like Large Language Models, intricate and novel connections form a bridge that connects users to the essence of the Mi'kmaq community and an Ancient Indigenous Language.

            A crucial component of the :red[Lnu-AI] project is our emphasis on cultural sensitivity and accuracy. :red[Lnu-AI] respects and accurately portrays the intricacies of the Mi'kmaq language and culture. :red[Lnu-AI] offers a technologically advanced, culturally respectful, and accurate platform, fostering a genuine understanding of the Mi'kmaq heritage.

            The:red[Lnu-AI] Project, seeks to preserve, promote, and celebrate the beauty of the Mi'kmaq language through Storytelling, Conversation and offering a deeper insight into this beautiful ancient language.
            """)

    elif page_selection == "Helpful Links":
        st.subheader('Helpful Links')
        st.markdown("""
            - [Assembly of First Nations](http://www.afn.ca/)
            Discover the Assembly of First Nations (AFN), Canada's premier national representative organization of over 630 First Nation communities.

            - [Atlantic Canada's First Nation Help Desk](http://firstnationhelp.com/)
            Visit Atlantic Canada's First Nation Help Desk, promoting digital literacy among students and teachers through accessible Internet content.

            - [Listuguj Mi'gmaq Government](https://listuguj.ca/)
            Explore the official portal of the Listuguj Mi'gmaq Government for comprehensive information on community, governance, and services.

            - [Passamaquoddy-Maliseet Language Portal](http://pmportal.org/)
            Connect with the rich linguistic heritage of Passamaquoddy-Maliseet speakers through the Language Portal's comprehensive dictionary and video archives.

            - [Mi'gmawei Mawio'mi](http://www.migmawei.ca)
            Learn about the Mi'gmawei Mawiomi Secretariat (Tribal Council), a collective representative body founded in 2000 for the Councils of Gespeg, Gesgapegiag, and Listuguj.

            - [Mi'kmaq Resource Centre](https://www.cbu.ca/indigenous-affairs/unamaki-college/mikmaq-resource-centre/)
            Engage with the rich repository of documents at the Mi'kmaq Resource Centre, a part of Unama'ki College at Cape Breton University, dedicated to Aboriginal research.

            - [Mi’gmaq-Mi’kmaq Online](https://www.mikmaqonline.org/)
            Access the Mi’gmaq-Mi’kmaq Online Dictionary, a comprehensive resource for the Mi’gmaq language, featuring a searchable database of words, phrases, and audio files.
            
            - [Native Languages of the Americas: Mi'kmaq](http://www.native-languages.org/mikmaq.htm)
            Experience the dedicated work of the non-profit organization, Native Languages of the Americas, in preserving and promoting indigenous languages including Mi'kmaq.

            - [NativeTech: Native American Technology & Art](http://www.nativetech.org/)
            Dive into the world of indigenous ethno-technology with NativeTech, focusing on the historical and contemporary arts of Eastern Woodland Indian Peoples.

            - [`Nnui Tli'suti/`Lnuei Tli'suit](http://nnueitlisuti.webs.com/)
            Access Watson Williams' detailed lesson plans on Mi'gmaw language, featuring reading and writing exercises in both the Listuguj and Smith-Francis writing systems.
        """)
        st.info("These links and more will be converted into an Vector Embedded database to preserve the content for future generations.")


def main_application(global_vars: dict) -> None:
    """
    Main application function.

    Args:
        global_vars (dict): A dictionary containing all global variables.
    """
    all_word_details = global_vars['all_word_details']
    tts_settings = global_vars['tts_settings']
    
    # Custom CSS for the image and title
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    

    # Define file path for "bnr1.png"
    image_path = os.path.join(os.getcwd(), "images", "bnr1.png")

    # Load and display the image
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True, clamp=True, channels="RGB", output_format="png")

    # Instruction text
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <div class="big-font -text center-text" style="color: crimson; font-size: 24px;">Generate a Story</div>
        </div>
        <ul>
            <li>Type an English word into the 'Search for a word' field.</li>
            <li>Select a Mi'kmaq word from the 'Similar words' list.</li>
            <li>Click Generate Story and an audio and visual story will be generated.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    search_word = st.sidebar.text_input("Search for a word", value="")

    # Initialize sidebar
    sidebar = st.sidebar

    # Center the image using Streamlit's layout feature
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=None, clamp=True, channels="RGB", output_format="PNG")

    # Instruction text
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="big-font">
                <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Storyteller</span>
                <span style="color:white;">:</span>
                <span style="font-size:24px; color:crimson;">A'tugwewinu</span>
            </div>
            <ul style="text-align: left;">
                <li>Mi'kmaq storytelling plays a role in preserving the Mi'kmaq language.</li>
                <li>Storytelling is a vital aspect of the Mi'kmaq people's culture and tradition.</li>
                <li>It serves as a means to preserve and transmit knowledge, values, beliefs, and history from one generation to the next.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Initialize the variable selected_word
    selected_word = None

    # Load word data
    if all_word_details is None:
        st.sidebar.error("Failed to load word data.")
        return

    # Generate word list and transform data to DataFrame
    word_list = list(all_word_details.keys())
    df = pd.DataFrame([(key, value["translation"], value["meanings"], value["example"]) for key, value in all_word_details.items()],
                        columns=["word", "translation", "meanings", "example"])

    # Sort the DataFrame by 'word' column in ascending order
    df = df.sort_values(by=['word'])

    # Resetting the index
    df = df.reset_index(drop=True)

    similar_words = None
    selected_word = ""
    user_selected_word = ""

    # Check if a search_word has been entered
    if search_word:  
        search_word_lower = search_word.lower()

        # Find the exact match in the "translation" field where the searched word is the first word
        exact_match_translation = df[df['translation'].apply(lambda translation: translation.lower().split()[0] == search_word_lower)]['word'].values

        if len(exact_match_translation) > 0:
            selected_word = exact_match_translation[0]
        else:
            # Find the word in the "meanings" field where the searched word is present
            similar_words_meanings = df[df['meanings'].apply(lambda meanings: any(search_word_lower in meanings.lower().split() for meanings in meanings))]['word'].values
            if len(similar_words_meanings) > 0:
                selected_word = similar_words_meanings[0]

        if not selected_word:
            st.sidebar.write("No similar word found.")
        
        if selected_word:
            # Get index of the selected word in dataframe
            selected_word_index = df.index[df['word'] == selected_word].tolist()[0]

            # Get next 19 words after the selected word
            next_words = df.iloc[selected_word_index+1 : selected_word_index+20]['word'].values

            # Combine the selected word with next 19 words
            combined_words = np.concatenate(([selected_word], next_words))

            user_selected_word = st.sidebar.selectbox("Similar words:", combined_words)

    # Display word details in sidebar
    if selected_word:
        display_word_details_main(selected_word, all_word_details, tts_settings, sidebar)

        # TTS service selection in the sidebar
        tts_service = sidebar.selectbox("Select a TTS service", ['gtts'], key="tts_service_selectbox", index=0)
        tts_settings["tts_audio"] = tts_service if tts_service else 'gtts'

        # Display selected word below submit button
        st.sidebar.markdown(f"Selected word: **{selected_word}**")
        
        # Submit button in the sidebar
        submit_button = sidebar.button(
            f"Generate Story about **{selected_word}**",
            help="Click to generate the story",
            key="submit_button",
            args=(selected_word, all_word_details, tts_service),
            kwargs={"generate_images": selected_word == "Submit with Images"},
            type="primary"
        )

        # Generate and display story, audio, and images
        if submit_button:
            st.info("Generating the story... This may take a minute. It's worth the wait!")  # <--- Display info message
            jsonl_file = "mikmaq_semantic.jsonl"
            themes = load_theme_and_story(jsonl_file)
            word_details = get_user_inputs(selected_word, all_word_details)
            if word_details is not None:
                meaning = word_details.get('meanings', [])[0] if word_details.get('meanings') else ""
                theme, story_word, image_theme = get_theme_and_story_word(themes, selected_word, meaning)
                story_text, _, _ = generate_story(word_details, theme, story_word, image_theme)

                try:
                    audio = generate_audio(story_text, tts_service)
                    display_story_and_audio(story_text, audio)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                generate_and_display_images(story_text, image_theme)
                st.empty()  # <--- Removes info message


@st.cache_data
def get_tts_service(tts_settings: dict) -> str:
    """
    Function to determine the appropriate Text-to-Speech service based on the settings.

    Args:
        tts_settings (dict): Text-to-Speech settings.

    Returns:
        str: Name of the Text-to-Speech service.
    """
    tts_service = next((service for service, flag in tts_settings.items() if flag.lower() == 'yes'), 'gtts')
    return tts_service



def get_theme_and_story_word(themes: Optional[List[Dict[str, str]]], selected_word: str, meaning: str) -> Tuple[str, str, str]:
    """
    Function to handle theme and story word selection.
    """
    if not themes:
        raise ValueError("No themes provided.")

    selected_theme = select_random_theme(themes)
    return replace_placeholders_in_theme(selected_theme, selected_word, meaning)



def select_random_theme(themes: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Selects a random theme from the provided list.

    Args:
        themes (List[Dict[str, str]]): The list of themes to choose from.

    Returns:
        Dict[str, str]: The selected theme.
    """
    return random.choice(themes)



def replace_placeholders_in_theme(theme: Dict[str, str], word: str, meaning: str) -> Tuple[str, str, str]:
    """
    Replaces placeholders in a theme with the provided word and meaning.

    Args:
        theme (Dict[str, str]): The theme.
        word (str): The word to replace the '{word}' placeholder.
        meaning (str): The meaning to replace the '{meaning}' placeholder.

    Returns:
        Tuple[str, str, str]: A tuple containing the theme, story word, and image theme.

    Raises:
        KeyError: If a required key is missing from the theme.
    """
    try:
        theme_text = theme['theme'].replace('{word}', word).replace('{meaning}', meaning)
        story_word = theme['story_word']
        image_theme = theme['image_theme'].replace('{word}', word).replace('{meaning}', meaning)
    except KeyError as e:
        raise KeyError(f"Required key missing from theme: {str(e)}")

    return theme_text, story_word, image_theme



def generate_story(all_word_details: dict, theme: str, story_word: str, image_theme: str) -> Tuple[str, str, str]:
    """
    Function to generate a story using OpenAI's GPT-4 model. Interacts with the OpenAI API to create a conversation
    and uses the returned message content as the generated story.

    Args:
        all_word_details (dict): Dictionary of all word details.
        theme (str): The theme for the story.
        story_word (str): The main word for the story.
        image_theme (str): The theme for the image.

    Returns:
        str: The generated story text.
        str: The main word for the story.
        str: The theme for the image.
    """

    # The model used for generating the story is retrieved from environment variables.
    # If there is no such variable defined, use "gpt-4-0613" as the default model.
    # You may want to replace this with a different model name if a new version is available.
    
    model = os.getenv("CHAT_MODEL_SELECTION", "gpt-4-0613")

    if model == "gpt-4-0613":
        st.info("Environment variable for MODEL_SELECTION is not set, using default model: gpt-4-0613")

    # Define the system's role and content. The system plays the role of a Mi'kmaq storyteller.
    prompt_system = {
        'role': 'system',
        'content': "You are a Mi'kmaq storyteller, or an 'a'tugwewinu', skilled in weaving captivating tales about the Mi'kmaq people and their rich cultural heritage."
    }

    # Define the initial part of the story as an 'Assistant' message.
    initial_story = {
        'role': 'assistant',
        'content': f"Let us embark on a journey with a theme of '{theme}', centered around the word '{story_word}'."
    }

    # Create a conversation with OpenAI's Chat models.
    # Parameters like max_tokens, temperature, top_p, frequency_penalty, and presence_penalty can be tweaked
    # to adjust the output from the model.
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[prompt_system, initial_story],
        max_tokens=2000,  # Maximum length of the generated text. Consider adjusting this for longer/shorter outputs.
        temperature=0.5,  # Controls the randomness of the output. Higher values (closer to 1) make output more random.
        top_p=1.0,  # Controls the nucleus sampling. Lower values can make the output more focused.
        frequency_penalty=0.5,  # Controls penalizing new tokens based on their frequency.
        presence_penalty=0.5,  # Controls penalizing new tokens based on their presence.
    )

    # This is the generated story text.
    story_text = response['choices'][0]['message']['content']

    # Return the generated story, story word, and image theme.
    return story_text, story_word, image_theme


def process_story_generation(selected_word: str, all_word_details: dict, generate_images: bool, tts_settings: dict) -> None:
    """
    Function to generate and display a story, its audio, and images.

    Args:
        selected_word (str): The selected word for the story.
        all_word_details (dict): Dictionary of all word details.
        generate_images (bool): Flag to decide if images need to be generated.
        tts_settings (dict): Text-to-Speech settings.
    """
    if not selected_word:
        st.info("Please enter a word for the story")
        return

    with st.spinner("Generating story..."):
        try:
            word_detail = all_word_details[selected_word]
            meaning = word_detail.get('meanings', [])[0]  # get the first meaning
            themes = load_theme_and_story("mikmaq_semantic.jsonl")
            theme, story_word, image_theme = get_theme_and_story_word(themes, selected_word, meaning)
            story_text, _, _ = generate_story(selected_word, theme, story_word, image_theme)

            tts_service = get_tts_service(tts_settings)
            audio = generate_audio(story_text, tts_service)
            display_story_and_audio(story_text, audio)

            st.success("Story generation completed!")
        except Exception as e:
            st.error(f"Error in generating story or audio: {str(e)}")

    if generate_images:
        generate_and_display_images(story_text, image_theme)


def display_story_and_audio(story_text: str, audio: Optional[str]) -> None:
    """
    Function to display the story text and its audio.

    Args:
        story_text (str): The generated story text.
        audio (Optional[str]): Path to the audio file.
    """
    story_container = st.container()
    with story_container:
        st.markdown(
            f"<div style='background-color: #2f4f4f; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);'>"
            f"<p style='font-size: 18px; font-weight: bold; margin-bottom: 10px;'>Story text:</p>"
            f"<p style='font-size: 16px; line-height: 1.5;'>{story_text}</p>"
            "</div>",
            unsafe_allow_html=True
        )
        if audio is not None and os.path.isfile(audio):
            with open(audio, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            os.remove(audio)  # Delete the temporary audio file after playing


def generate_and_display_images(story_text: str, image_theme: str) -> None:
    """
    Function to generate and display images for different parts of the story.

    Args:
        story_text (str): The generated story text.
        image_theme (str): The theme for the image.
    """
    story_parts = [
        story_text[:len(story_text) // 3],
        story_text[len(story_text) // 3: 2 * len(story_text) // 3],
        story_text[2 * len(story_text) // 3:],
    ]

    image_container = st.container()
    for i, part in enumerate(story_parts):
        image_url = generate_openai_images(part + ' ' + image_theme)
        if image_url:
            image_container.image(image_url, width=None, clamp=True, channels="RGB", output_format="png") # image_container used directly
        else:
            image_container.markdown("**Image generation failed.**") # image_container used directly


def get_user_inputs(selected_word: str, all_word_details: dict) -> Optional[Dict]:
    """
    Function to validate and return word details based on user inputs.
    
    Args:
        selected_word (str): The word selected by the user.
        all_word_details (dict): Dictionary of all word details.
        
    Returns:
        Optional[Dict]: Details of the selected word, None if not present.
    """
    # Check if selected_word is in all_word_details
    if selected_word not in all_word_details:
        st.error(f"The word {selected_word} does not exist in the dictionary.")
        return None
    return all_word_details[selected_word]


def display_word_details_main(selected_word: str, all_word_details: dict, tts_settings: dict, sidebar) -> None:
    """
    Function to display the details of a selected word.

    Args:
        selected_word (str): The word selected by the user.
        all_word_details (dict): Dictionary of all word details.
        tts_settings (dict): Text-to-Speech settings.
        sidebar (Streamlit Sidebar): The sidebar object for output.
    """
    word_detail = get_user_inputs(selected_word, all_word_details)
    if word_detail is None:
        return

    # Display word details
    sidebar.markdown(f"<h3 style='color: crimson;'><span style='font-size: 28px; font-weight: bold;'>{selected_word}</span></h3>", unsafe_allow_html=True)  # changed to sidebar
    sidebar.markdown(f"<div style='color:CornflowerBlue'>Pronunciation guide: {word_detail.get('pronunciation', '')}</div>", unsafe_allow_html=True)  # changed to sidebar
    sidebar.markdown(f"Part of speech: {word_detail.get('part_of_speech', '')}")  # changed to sidebar
    
    # Display meanings
    meanings = word_detail.get('meanings', [])
    sidebar.markdown("Meanings:")  # changed to sidebar
    for meanings in meanings:
        sidebar.markdown(f"- {meanings}")  # changed to sidebar

    # Display example sentences
    example_sentences = word_detail.get('example', [])
    sidebar.markdown("Example of word used in sentence:")  # changed to sidebar
    for example_sentence in example_sentences:
        sidebar.markdown(f"- {example_sentence}")  # changed to sidebar

    # Display pronunciation
    sidebar.markdown("Listen to pronunciation:")  # changed to sidebar
    tts_service = tts_settings.get('tts_audio', 'gtts')
    audio = generate_audio(selected_word, tts_service)
    if os.path.isfile(audio):
        audio_file = open(audio, 'rb')
        audio_bytes = audio_file.read()
        sidebar.audio(audio_bytes, format='audio/wav')  # changed to sidebar
        os.remove(audio)  # Delete the temporary audio file after playing


def main():
    """
    The main function of the application.
    """
    global_vars = load_env_variables_and_data()

    # You can use the global_vars dictionary to access the required values.
    api_keys = global_vars['api_keys']
    tts_settings = global_vars['tts_settings']
    local_data_files = global_vars['local_data_files']
    models = global_vars['models']
    completion_model = global_vars['completion_model']
    all_word_details = global_vars['all_word_details']
    
    trained_data = global_vars['trained_data']
    trained_data_embeddings = global_vars['trained_data_embeddings']
    word_details_embeddings = global_vars['word_details_embeddings']
    
    render_ui(CUSTOM_CSS)

    menu_options = {
        "Storyteller": lambda: main_application(global_vars),
        "Lnu-AI Chat": lambda: chatbot_application(models, trained_data, local_data_files, tts_settings),
        "Examine Words": explore_words,
        "Language Preservation": display_mikmaq_resources,
    }

    selected_option = render_menu(list(menu_options.keys()))

    if selected_option and menu_options.get(selected_option):
        menu_options[selected_option]()


def render_ui(CUSTOM_CSS):
    """
    Renders the user interface components.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: Crimson; margin-top: -70px;">Lnu-AI</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">An Indigenous AI System</h3>', unsafe_allow_html=True)
    st.divider()


def render_menu(options: list) -> str:
    """
    Renders the menu with options.

    Args:
        options (list): A list of options to be displayed in the menu.

    Returns:
        str: The selected option from the menu.
    """
    icons = ["book", "chat", "puzzle fill", "archive"]
    return option_menu(None, options, icons=icons, menu_icon="cast", default_index=0, orientation="horizontal")


if __name__ == "__main__":
    main()
#wela'lin