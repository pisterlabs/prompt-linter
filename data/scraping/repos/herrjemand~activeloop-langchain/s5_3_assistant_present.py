from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

chroma_dataset_name = "assistant_53"
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
vecdb = Chroma(
    client=chroma_client,
    collection_name=chroma_dataset_name,
    embedding_function=embeddings,
)


retrieval_qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vecdb.as_retriever(
        distance_metric="cos",
        fetch_k=100,
        maximal_marginal_relevance=True,
        k=4,
    ),
)


import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain.chains import RetrievalQA
from streamlit_chat import message
from openai import Audio

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            response = Audio.transcribe("whisper-1", f)

        return response["text"]
    
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
    
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
        
    else:
        st.write("No transcription available.")

def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")

def search_db(user_input):
    return retrieval_qa({"query": user_input})

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], is_user=False, key=f"{i}_bot")

        voice = "Bella"
        text = history["generated"][i]
        audio = generate(text, voice=voice)
        st.audio(audio, format="audio/mp3")

st.write("# Assistant 5.3")
transcript = record_and_transcribe_audio()
user_input = get_user_input(transcript)

if "generated" not in st.session_state:
    st.session_state.generated = ["I am ready to help you"]

if "past" not in st.session_state:
    st.session_state.past = ["Hey there"]

if user_input:
    output = search_db(user_input)
    print(output)

    st.session_state.past.append(user_input)
    response = str(output["result"])
    st.session_state.generated.append(response)

# if st.session_state.generated:
display_conversation(st.session_state)