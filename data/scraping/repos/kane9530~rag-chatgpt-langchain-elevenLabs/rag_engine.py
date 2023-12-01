import os, tempfile
import requests 
from pathlib import Path
import subprocess
import pygame.mixer
from gtts import gTTS
from io import BytesIO

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

import pinecone
import openai
from pydub import AudioSegment
from pydub.playback import play

import streamlit as st
from audiorecorder import audiorecorder
import nltk
import ssl
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')
st.set_page_config(page_title="Ask Kane")
if 'process_documents_success' not in st.session_state:
    st.session_state['process_documents_success'] = False

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, 
                                     embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

@st.cache_resource
def init_memory():
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(openai_api_key=st.session_state.openai_api_key,
                    model_name='gpt-3.5-turbo'),
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
    
def get_chat_history(inputs):
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    print("\n".join(res))
    return "\n".join(res)

# Prompt
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question, add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


def query_llm(retriever, query):
    llm= ChatOpenAI(openai_api_key=st.session_state.openai_api_key,
                    model_name='gpt-3.5-turbo')
    memory = init_memory()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        get_chat_history=get_chat_history, 
        verbose=True

    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    print(st.session_state.messages)
    return result

def check_secrets():
    # Checks bar if the relevant api_key is not present in the .streamlit/secrets.toml file.
    if "openai_api_key" in st.secrets.openai:
        print("openai api key found!")
        st.session_state.openai_api_key = st.secrets.openai.openai_api_key
        os.environ["OPENAI_API_KEY"] = st.secrets.openai.openai_api_key
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        print("openai api key NOT found!")
    if "pinecone_api_key" in st.secrets.pinecone:
        print("pinecone api key found!")
        st.session_state.pinecone_api_key = st.secrets.pinecone.pinecone_api_key
        os.environ["PINECONE_API_KEY"] =  st.secrets.pinecone.pinecone_api_key
    else: 
        print("pinecone api key NOT found!")
    if "pinecone_env" in st.secrets.pinecone:
        print("pinecone env key found!")
        st.session_state.pinecone_env = st.secrets.pinecone.pinecone_env
        os.environ["PINECONE_ENV"] =  st.secrets.pinecone.pinecone_env
    else:
        print("pinecone env key NOT found!")
    if "pinecone_index" in st.secrets.pinecone:
        print("pinecone index found!")
        st.session_state.pinecone_index = st.secrets.pinecone.pinecone_index
        os.environ["PINECONE_INDEX"] =  st.secrets.pinecone.pinecone_index
    else:
        print("pinecone index key NOT found!")
    if "eleven_labs_api_key" in st.secrets.elevenlabs:
        print("eleven labs api key  found!")
        os.environ["ELEVEN_LABS_API_KEY"] =  st.secrets.elevenlabs.eleven_labs_api_key
    else:
        print("eleven labs api key NOT found!")
        
    if "advisor_voice_id" in st.secrets.elevenlabs:
        print("eleven labs advisor_voice_id found!")
        os.environ["ELEVEN_LABS_ADVISOR_KEY"] =  st.secrets.elevenlabs.advisor_voice_id
    else:
        print("eleven labs advisor key NOT found!")


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                documents = load_documents()
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                texts = split_documents(documents)
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                    st.session_state.process_documents_success = True
                    print(f"In local db:{st.session_state.process_documents_success}")
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
                    st.session_state.process_documents_success = True
                    print(f"In pinecone db:{st.session_state.process_documents_success}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
def get_text_from_speech(audio_file):
    audio_file = open("audio.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    return transcript['text']

def get_speech_from_text(text):
    advisor_key=os.getenv("ELEVEN_LABS_ADVISOR_KEY")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{advisor_key}/stream"
    data = {
        "text": text.replace('"', ''),
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.3
        }
    }
    
    r = requests.post(url, headers={'xi-api-key':os.getenv("ELEVEN_LABS_API_KEY")}, json=data)
    output_file_path = os.path.join("/tmp", "reply.mp3")
    with open(output_file_path, "wb") as output:
        output.write(r.content)
    pygame.mixer.init()
    # Load and play the audio file
    pygame.mixer.music.load(output_file_path)
    pygame.mixer.music.play()
    print('playing sound using pygame')
            

def boot():
    #
    check_secrets()
    st.chat_message('ai').write("Hi, This is Kane. How can I help you today?")
    with open("styles.css", "r") as css_file:
            styles = css_file.read()
            st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True) 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.markdown("# 1) Upload Document")
        st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
        image = st.image("images/kane.png", caption="Kane, your friendly AI bot", use_column_width=True)
        st.session_state.source_docs = st.file_uploader(label="Upload Files", type="pdf", accept_multiple_files=True)
        submit_button = st.button("Submit Document", on_click=process_documents, type="primary")
        st.markdown("---")
# Smaller sidebar on the left below panel 1
    if st.session_state.process_documents_success: 
        st.sidebar.markdown("# 2) Ask Kane", unsafe_allow_html=True)
        with st.sidebar:
            audio = audiorecorder("Click to record", "Click to stop recording")

        if not audio.empty():
            audio_duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            if audio_duration < 1:
                st.sidebar.warning('Press the icon above and start a conversation', icon="⚠️")
            else:
                st.sidebar.audio(audio.export().read())  
                audio.export("audio.wav", format="wav") 
                transcript = get_text_from_speech("audio.wav")
                response = query_llm(st.session_state.retriever, transcript)
                sound = BytesIO()
                tts = gTTS(response, lang='en', tld='com')
                tts.write_to_fp(sound)
                
                #response_audio = get_speech_from_text(response)
                for message in st.session_state.messages:
                    st.chat_message('human').write(message[0])
                    st.chat_message('ai').write(message[1])
                    st.audio(sound)

if __name__ == '__main__':
    boot()