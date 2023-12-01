import os
import tempfile

import speech_recognition as sr
import streamlit as st
import telebot
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from pydub import AudioSegment

# Load the environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
bot = telebot.TeleBot(BOT_TOKEN)


def load_correction_chain(llm):
    prompt_template = "Correct any grammatical errors in the following text:\n{text}"
    correction_prompt = PromptTemplate(
        template=prompt_template, input_variables=["text"])

    return llm.load_chain(chain_type="map_reduce", map_prompt=correction_prompt)


def load_paraphrase_chain(llm):
    prompt_template = "Generate 5 diverse paraphrase of the following text:\n{text}"
    paraphrase_prompt = PromptTemplate(
        template=prompt_template, input_variables=["text"])

    return llm.load_chain(chain_type="map_reduce", map_prompt=paraphrase_prompt)


@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    input_text = message.text
    first_word = input_text.split()[0].lower()

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(input_text)

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]

    if first_word == "correction":
        # Grammar correction
        chain = load_correction_chain(llm, chain_type='map_reduce')
    elif first_word == "paraphrase":
        # Paraphrasing
        chain = load_paraphrase_chain(llm, chain_type='map_reduce')
    elif first_word == "summary":
        # Text summarization
        chain = load_summarize_chain(llm, chain_type='map_reduce')
    else:
        bot.reply_to(
            message, "Please start your message with either 'correction', 'paraphrase' or 'summary'.")
        return

    response = chain.run(docs)
    bot.reply_to(message, response)


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    # Download the voice message file
    voice_file = bot.get_file(message.voice.file_id)

    # Create a temporary file to save the voice message
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name

    # Download the voice message file to the temporary location
    downloaded_file = bot.download_file(voice_file.file_path)

    # Save the downloaded file to the temporary location
    with open(file_path, 'wb') as f:
        f.write(downloaded_file)

    # Convert the audio file to the WAV format using pydub
    audio = AudioSegment.from_file(file_path, format="ogg")
    wav_file_path = file_path + ".wav"
    audio.export(wav_file_path, format="wav")

    # Convert the voice file to text using speech_recognition
    r = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)

    # Pass the text to the existing message handler function
    message.text = text
    echo_all(message)


bot.infinity_polling(timeout=10, long_polling_timeout=5)
