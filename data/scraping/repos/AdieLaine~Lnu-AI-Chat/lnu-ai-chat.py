# System-related
import os
import tempfile
import time

# File and IO operations
import json

import soundfile as sf

# Data processing and manipulation
import pandas as pd
import random

# Image processing
from PIL import Image


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
    page_title="Lnu-AI Chat - An Indigenous AI System",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/Lnu-AI-Chat',
        'Report a bug': 'https://github.com/AdieLaine/Lnu-AI-Chat/issues',
        'About': """
            # Lnu-AI
            Welcome to Lnu-AI Chat! This application is dedicated to helping people learn and appreciate the Mi'kmaq language, an indigenous language of Eastern Canada and the United States. 

            ## About Mi'kmaq Language
            The Mi'kmaq language is a rich, polysynthetic language with a deep historical and cultural significance. It is, however, at risk of being lost as the number of fluent speakers decreases.

            ## The Lnu-AI Project
            Lnu-AI utilizes advanced AI technologies to provide a platform for learning, using, and preserving the Mi'kmaq language. It offers various features like chat functionality, storytelling, and deep linguistic analysis to facilitate language learning and appreciation.

            ## Your Contribution
            As an open-source project, we welcome contributions that help improve Lnu-AI and further its mission to preserve the Mi'kmaq language. Please visit our [GitHub](https://github.com/AdieLaine/Lnu-AI-Chat) page for more information.
            
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


def get_env_variable(name, default=None):
    """
    Returns the value of the environment variable with the given name.
    First tries to fetch it from Streamlit secrets, and if not available,
    falls back to the local environment. If it's not found in either place,
    returns the default value if provided.
    
    Args:
        name (str): The name of the environment variable.
        default (str, optional): The default value to be returned in case the environment variable is not found.

    Returns:
        str: The value of the environment variable, or the default value.
    """
    if st.secrets is not None and name in st.secrets:
        # Fetch the secret from Streamlit secrets
        return st.secrets[name]
    else:
        # Try to get the secret from the local environment
        value = os.getenv(name)

        if value is None and default is not None:
            # If the environment variable is not found and a default value is provided,
            # return the default value
            return default
        else:
            return value


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
        "openai": get_env_variable("OPENAI_API_KEY"),
    }

    tts_settings = {
        "tts_audio": get_env_variable('TTS_AUDIO'),
        "local_tts": get_env_variable('LOCAL_TTS'),
    }

    local_data_files = {
        "trained_data": (os.getenv("TRAINED_DATA") + '.jsonl') if os.getenv("TRAINED_DATA") else None,
        "all_word_details": (os.getenv("ALL_WORD_DETAILS") + '.json') if os.getenv("ALL_WORD_DETAILS") else None,
        "trained_data_embeddings": (os.getenv("TRAINED_DATA_EMBEDDINGS") + '.json') if os.getenv("TRAINED_DATA_EMBEDDINGS") else None,
        "word_details_embeddings": (os.getenv("WORD_DETAILS_EMBEDDINGS") + '.json') if os.getenv("WORD_DETAILS_EMBEDDINGS") else None,
    }

    models = {
        "chat_model": os.getenv("CHAT_MODEL_SELECTION", default="gpt-3.5-turbo-0613"),
        "completion_model": os.getenv("COMPLETION_MODEL_SELECTION", default="gpt-3.5-turbo-0613"),
        "fine_tuned_model_dictionary": os.getenv("FINE_TUNED_MODEL_DICTIONARY"),
        "fine_tuned_model_data": os.getenv("FINE_TUNED_MODEL_DATA")
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
            {"role": "system", "content": get_system_message()},  # Provides assistant directives
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_reply}
        ]

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=gpt_messages,
            max_tokens=1200,
            temperature=0.4,
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
        # Log the error for later debugging
        log_error(f"Error with {model_name}: {e}")
        raise e  # Rethrow the exception after logging

    # If there was no error but also no 'choices' from OpenAI
    return "I didn't quite get that. Could you rephrase or provide more details, please?"


def get_system_message():
    # Update this to generate a system message dynamically based on your use case.
    return "You are Lnu-AI, an AI system developed to promote and preserve Mi'kmaq language and culture."


def log_error(message):
    # Update this to log the error message in a way that's appropriate for your application.
    print(message)


fallback_responses = [
    #The use of varied responses allows the assistant to provide a more natural conversation experience.
    "I'm having a bit of trouble understanding your request. Could you please provide more details or try phrasing it differently?",
    "Could you please clarify your question?",
    "It seems I'm unable to assist with your query. Could you ask in a different way?",
    "I'm currently having difficulty comprehending that. Could you offer more context or details?",
]


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

    # If no match found in trained data, generate completion with the trained models, enhanced as a fine-tuned model.
    model_to_use = models["fine_tuned_model_data"] if "fine_tuned_model_data" in models else models.get("completion_model", "curie:ft-personal-2023-05-16-05-11-43")
    
    response = openai.Completion.create(
        model=model_to_use,
        prompt=prompt,
        max_tokens=1200,
        temperature=0.4,
    )
    
    if 'choices' in response and len(response['choices']) > 0:
        if 'text' in response['choices'][0]:
            final_reply = response['choices'][0]['text']
            final_reply = clean_reply(final_reply)
            
            # Enhance the reply using GPT
            final_reply = enhance_with_gpt(prompt, final_reply, models)

            # Prompt-reply pair compared to the trained_data
            new_prompt_data = {'prompt': prompt, 'completion': final_reply}
            trained_data.append(new_prompt_data)

            return final_reply
    
    # Default response if no response is generated
    return random.choice(fallback_responses)


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
def display_word_details_chatbot(selected_word, all_word_details: dict):
    """
    Displays details of a selected word in the chatbot interface.

    Parameters:
        selected_word (str): Selected word.
        all_word_detail (dict): Details of all words.
    """
    if selected_word not in all_word_details:
        st.error(f"The word {selected_word} does not exist in the dictionary.")
        return

    word_detail = all_word_details[selected_word]
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


def chatbot_application(models, trained_data, tts_settings):
    """
    The main chatbot application function.

    This function handles loading previous session states or initializes new ones, sets up the sidebar and the main chat area,
    and processes user input and chatbot responses.

    Parameters:
        models (dict): A dictionary of the trained models.
        trained_data (dict): A dictionary of the trained data.
        tts_settings (dict): Text-to-speech settings.
    """

    # Load previous session states or initialize new ones
    st.session_state.setdefault('past', [])
    st.session_state.setdefault('generated', [])

    # Add custom CSS for the image and title
    st.markdown("""
        <style>
        .explore-words-title {
            font-size: 24px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Define file path for image and load it
    image_path = os.path.join(os.getcwd(), "images", "chatside.png")
    image = Image.open(image_path)

    # Set up sidebar
    st.sidebar.image(image, use_column_width="auto", clamp=True, channels="RGB", output_format="png")
    st.sidebar.markdown("---")
    topics = [
        "The Tale of Glooscap",
        "The Mi'kmaq Creation Story",
        "Tell me a Story",
        "Discuss Mi'kmaq Linguistics",
        "Explore Mi'kmaq History",
        "Culture and Traditions",
        "Let's Make a Canoe",
        "Craft a Sweetgrass Basket",
        "Create a Streamlit App",
        "Define Lnu-AI Features",
    ]

    sidebar_text = "\n- ".join(topics)
    sidebar_text = f"**Ask me about:**\n\n- {sidebar_text}\n\n**Note:** Audio responses will display here for each topic."

    st.sidebar.info(sidebar_text)

    # Instruction text
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="big-font">
                <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Talk Together</span>
                <span style="color:white;">:</span>
                <span style="font-size:24px; color:crimson;">Mawagnutmajig</span>
            </div>
            <ul style="text-align: left;">
                <li>I’m Lnu-AI, your Indigenous AI Assistant.</li>
                <li>I am still learning and won’t always get it right, but our conversations will help me improve.</li>
                <li>The sharing of words preserves and transmits knowledge, values, beliefs, and history from one generation to the next.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Placeholder for the audio player in the sidebar
    audio_placeholder = st.sidebar.empty()

    # Chat form for user input
    user_message = st.chat_input("Chat with Lnu-AI")

    if user_message:
        user_message = user_message.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        st.session_state['past'].append(user_message)
        with st.spinner("Referring to my Mi'kmaq Corpus ..."):
            try:  # Exception handling added here
                chat_response = lnu_ai_chat(user_message, trained_data, models, all_word_details)
                # Generate audio response
                tts_service = tts_settings.get('tts_audio', 'gtts')
                audio_file = generate_audio(chat_response, tts_service)
                audio_response = open(audio_file, "rb")
                st.session_state['audio'] = audio_response.read()
                os.remove(audio_file)
            except Exception as e:  # If error occurs, set response as error message
                chat_response = f"Sorry, an error occurred while processing your message: {str(e)}"
            finally:  # Always append the response, even if an error occurred
                st.session_state['generated'].append(chat_response)

    # Display the audio player in the sidebar
    if 'audio' in st.session_state:
        audio_placeholder.audio(st.session_state['audio'], format="audio/wav")
    
    # Define avatar image path
    assistant_image_path = os.path.join(os.getcwd(), "images", "lnuavatar.png")
    image = Image.open(image_path)

    # Display the chat history
    for i in range(len(st.session_state['past'])):
        user_message = st.session_state['past'][i]
        chat_response = st.session_state['generated'][i]
        if chat_response:  # Only display non-empty messages
            with st.chat_message("User"):
                st.write(user_message)
            with st.chat_message("Lnu-AI", avatar=assistant_image_path):
                st.write(chat_response)


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
    all_word_details = global_vars['all_word_details']

    render_ui(CUSTOM_CSS)

    menu_options = {
        "Lnu-AI Chat": lambda: chatbot_application(models, trained_data, tts_settings),
    }

    selected_option = render_menu(list(menu_options.keys()))

    if selected_option and menu_options.get(selected_option):
        menu_options[selected_option]()


@st.cache_data
def render_ui(CUSTOM_CSS):
    """
    Renders the user interface components.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: Crimson; margin-top: -70px;">Lnu-AI Chat</h1>', unsafe_allow_html=True)
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
    icons = ["chat"]
    return option_menu(None, options, icons=icons, menu_icon="cast", default_index=0, orientation="horizontal")

if __name__ == "__main__":
    main()
#wantaqo'ti