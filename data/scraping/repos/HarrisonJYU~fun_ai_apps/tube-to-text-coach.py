import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

# import assemblyai as aai
import langchain
import requests
import streamlit as st
from langchain import LLMChain
from langchain import hub
from langchain.cache import SQLiteCache
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.llms import Clarifai
from md2pdf.core import md2pdf

PROJECT_NAME = 'tube-to-text-coach'

CLF_OPENAI_USER_ID = 'openai'
CLF_CHAT_COMPLETION_APP_ID = 'chat-completion'
CLF_GPT4_MODEL_ID = 'GPT-4'
CLF_GPT35_MODEL_ID = 'GPT-3_5-turbo'

CLARIFAI_PAT = st.secrets['CLARIFAI_PAT']

# aai.settings.api_key = st.secrets['ASSEMBLYAI_API_KEY']

cache_name = 'tube-to-text-cache'
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def check_video_url():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    response = requests.get(checker_url)
    return response.status_code == 200


def extract_youtube_video_id():
    # Regular expression to match YouTube video IDs
    pattern = r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)"
    return match.group() if (match := re.search(pattern, youtube_link)) else None


def get_video_name():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    response = requests.get(checker_url)
    return response.json()['title']


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def remove_local_dir(local_dir_path):
    print(f'Removing local files in {local_dir_path}')
    shutil.rmtree(local_dir_path, ignore_errors=True)


def load_audio():
    loader = YoutubeAudioLoader([youtube_link], st.session_state.save_dir)
    list(loader.yield_blobs())

routine_extractor_prompt = hub.pull("aaalexlit/sport-routine-to-program")
routine_extractor_prompt_short = hub.pull("aaalexlit/sport-routine-to-program-short")

def generate_routine():
    # Initialize a Clarifai LLM
    clarifai_llm = Clarifai(
        pat=CLARIFAI_PAT,
        user_id=CLF_OPENAI_USER_ID,
        app_id=CLF_CHAT_COMPLETION_APP_ID,
        model_id=CLF_GPT4_MODEL_ID,
    )
    # Create LLM chain
    if st.session_state.short_version:
        prompt = routine_extractor_prompt_short
    else:
        prompt = routine_extractor_prompt
    llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)
    with st.spinner('Generating routine'):
        result = llm_chain.run(vid_name=vid_name, vid_text=vid_text)
        simulate_steam_response(result)
        return result


def simulate_steam_response(result):
    # Simulate stream of response with milliseconds delay
    message_placeholder = st.empty()
    full_response = ''
    for chunk in result.splitlines():
        full_response += f"{chunk}\n"
        time.sleep(0.1)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(f"{full_response}▌")
    message_placeholder.markdown(full_response)


@st.cache_data(show_spinner="Transcribing the video")
def transcribe_with_assembly(youtube_video_id):
    load_audio()
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(f'{st.session_state.save_dir}/{os.listdir(st.session_state.save_dir)[0]}')
    return transcript.text


@st.cache_data(show_spinner="Transcribing the video")
def transcribe_with_whisper(youtube_video_id):
    loader = GenericLoader(
        YoutubeAudioLoader([youtube_video_id], st.session_state.save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    st.write(f'docs num = {len(docs)}')
    return docs[0].page_content


st.set_page_config(
    page_title="Tube-to-Text Coach",
    page_icon=":muscle:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    with open('app_description.md') as descr:
        st.write(descr.read())
    # st.subheader('**Demo**')
    # st.video('https://www.youtube.com/watch?v=rj76EDbaOX4')
    with st.expander('👉 Next Steps:'):
        st.write('''
            - Make routine generating faster and more robust that can handle longer videos, with RAGs or agent frameworks.
            - Future: handle videos with only images but no follow along text with video-to-text?        
                ''')

with st.container():
    left_col, right_col = st.columns(spec=[0.4, 0.6], gap='medium')

    with left_col:
        with st.form('options'):
            youtube_link = st.text_input(value='https://www.youtube.com/watch?v=1a7URy4pLfw',
                                         label='Enter exercise video youtube link:',
                                         help='Any valid YT URL should work')

            if check_video_url():
                st.video(youtube_link)

            show_extracted_text = st.checkbox('Show video transcript', value=False)
            short_version = st.checkbox('Short version',
                                        help="Just list the exercises without any description",
                                        key='short_version',
                                        value=False)

            generate_button = st.form_submit_button("Generate my routine")

            if not check_video_url():
                st.warning('Please input a valid Youtube video link')
                st.stop()

    with right_col:
        if generate_button:
            vid_name = get_video_name()
            st.session_state.save_dir = f'vids/{get_hashed_name(vid_name)}'
            try:
                youtube_video_id = extract_youtube_video_id()
                # vid_text = transcribe_with_assembly(youtube_video_id)
                vid_text = transcribe_with_whisper(youtube_video_id)
                if show_extracted_text:
                    with st.expander('Video transcript'):
                        st.write(vid_text)

                generated_routine = generate_routine()

                exported_pdf = tempfile.NamedTemporaryFile()
                md2pdf(pdf_file_path=exported_pdf.name,
                       md_content=generated_routine)

                with open(Path(exported_pdf.name), 'rb') as pdf_file:
                    download_pdf_button = left_col.download_button('Export to PDF', pdf_file,
                                                                   file_name=f'{vid_name}.pdf')
                if download_pdf_button:
                    left_col.write(f'Downloaded {vid_name}.pdf')
            except Exception as e:
                st.error(e)
            finally:
                if 'save_dir' in st.session_state:
                    remove_local_dir(st.session_state.save_dir)
