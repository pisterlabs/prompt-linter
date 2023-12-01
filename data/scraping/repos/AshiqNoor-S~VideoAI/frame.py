import os
import streamlit as st
import asyncio
import yt_dlp
import whisper
import openai
from langchain.schema import Document
from langchain.document_transformers import DoctranTextTranslator
from txtai.pipeline import Summary, Textractor
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import re



st.title("VideoAI App")
st.write("Welcome to the VideoAI app! Upload a YouTube URL to get started.")


st.title("Instructions")
st.write("1. Enter a YouTube URL.")
st.write("2. Choose the desired actions.")
st.write("3. Click 'Process' to start.")


os.environ["OPENAI_API_KEY"] = st.secrets["key"]
os.environ["OPENAI_API_MODEL"] = "davinci-002"

def download(video_id: str) -> str:
    video_url = f'{video_id}'
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': f'audio/{video_id}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts,{'ffmpeg_location': "/usr/bin/ffmpeg"}) as ydl:
        error_code = ydl.download([video_url])
        if error_code != 0:
            raise Exception('Failed to download video')

    return f'audio/{video_id}.m4a'

def transcribe(file_path: str) -> str:
    transcription = whisper_model.transcribe(file_path, fp16=False)
    return transcription['text']

def text_summary(text, maxlength=None):
    # Create a summary instance
    summary = Summary()
    result = summary(text)
    return result

def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def extract_video_id(youtube_url):
    pattern = r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\?|$)"
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(0)
    else:
        return None


# Download and transcribe the YouTube video
st.sidebar.header("YouTube Video Transcription")
video_link = st.sidebar.text_input("Enter YouTube Video ID:")
video_id = extract_video_id(video_link)
if st.sidebar.button("Get Transcription"):
    with st.spinner("Loading..."):
        file_path = download(video_id)
        whisper_model = whisper.load_model("base.en")
        input_text = transcribe(file_path)
        st.session_state.input_text = input_text
        st.markdown("*Your Input Text*")
        
        def boxed_text(text):
            return f'<div style="padding: 4%; border: 2px solid blue; border-radius: 15px; margin:2%;">{text}</div>'
        st.title("Transcription")
        text_to_display = input_text
        boxed_text_html = boxed_text(text_to_display)
        st.markdown(boxed_text_html, unsafe_allow_html=True)
    

# Main Streamlit app
st.sidebar.header("Options")
choice = st.sidebar.selectbox("Choose an option", ["","Summarize Text", "Chat"])


if st.button("Save & Process"):
    with st.spinner("Processing"):
        # get pdf text
        raw_text = st.session_state.input_text

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(
            vectorstore)
            
elif choice == "Summarize Text":
    if st.button("Summarize"):
        if "input_text" in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("*Your Input Text*")
                st.info(st.session_state.input_text)
            with col2:
                st.markdown("*Summary Result*")
                result = text_summary(st.session_state.input_text)
                st.success(result)
        else:
            st.warning("Please enter text and save it first.")

elif choice == "Chat":
    with st.spinner("Loading..."):
        st.write(css, unsafe_allow_html=True)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        user_question = st.text_input("Chat:")
        if user_question:
            handle_userinput(user_question)
