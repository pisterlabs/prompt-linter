# Path: pages/01_whisper video transcription.py

import streamlit as st
from moviepy.editor import *
import tempfile
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import LLMChain, OpenAI
import openai
from deta import Deta

llm = OpenAI(temperature=0)

# Connect to Deta Base with your Data Key
deta = Deta(st.secrets["DETA_KEY"])

# Create a new database "example-db"
# If you need a new database, just use another name.
db = deta.Base("example-db")


def summarize(transcription):
    text_splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(transcription)
    print(len(texts))
    print(texts[0])
    docs = [Document(page_content=t) for t in texts]
    print(len(docs))

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(docs)


def tasks(transcription):
    text_splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(transcription)
    docs = [Document(page_content=t) for t in texts]
    print(docs)

    prompt_template = """Extract the possible tasks of the CONTEXT, 
    like in the following example:
    EXAMPLE:
    TASKS:
    * Create an object (assign to Mariana)
    * Assign the task to Pedro (assign to Pedro)
    * Recreate the design and image (assign to Josh)
    * Call the client (not yet assigned)
    CONTEXT:
    {text}
    TASKS:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # list comprehension to create a list of load_summarize_chain objects
    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True)

    # list comprehension to run each document through its corresponding chain

    tasks = [
        load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True).run(
            [doc]
        )
        for doc in docs
    ]

    prompt_template = (
        "Create a single list, remove duplicated tasks and summarize tasks: {text}?"
    )

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

    text = " ".join(tasks)

    result = llm_chain(text)
    return result["text"]


def save_audio_to_tmp(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file path
        temp_file_path = os.path.join(temp_dir, file.name)

        with open(temp_file_path, "wb") as f:
            f.write(file.read())

        # temp_output_file_path = os.path.join(temp_dir, "extracted_audio.mp3")
        # # Save the file into the temporary folder
        transcript = ""
        with open(temp_file_path, "rb") as f:
            transcript = openai.Audio.transcribe(
                file=f,
                model="whisper-1",
                response_format="text",
            )
        keyDb = db.put({"transcription": transcript, "media": file.name})
        return (transcript, keyDb["key"])


# extract_audio_from_video(temp_file_path, temp_output_file_path)


def save_video_to_tmp(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file path
        temp_file_path = os.path.join(temp_dir, file.name)

        temp_output_file_path = os.path.join(temp_dir, "extracted_audio.mp3")
        # Save the file into the temporary folder
        with open(temp_file_path, "wb") as f:
            f.write(file.read())

        extract_audio_from_video(temp_file_path, temp_output_file_path)

        transcript = "not ready"
        with open(temp_output_file_path, "rb") as f:
            transcript = openai.Audio.transcribe(
                file=f,
                model="whisper-1",
                response_format="text",
            )
            keyDb = db.put({"transcription": transcript, "media": file.name})
            return (transcript, keyDb["key"])


def extract_audio_from_video(video_file, output_audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(output_audio_file)


def app():
    st.title("Pixelspace Experiments")
    st.write("This is the transcription audio page.")


st.title("WhisperAI Transcription service")


audioFile = st.file_uploader("Upload a file", type=["mp4", "mp3"])

if audioFile is not None:
    with st.spinner("Whisper is processing your file..."):
        st.write(audioFile.type)
        if audioFile.type == "audio/mpeg":
            with st.expander("Audio"):
                st.audio(audioFile)
                (transcription, key) = save_audio_to_tmp(audioFile)
        else:
            audioFile.type == "video/mp4"
            with st.expander("Video"):
                st.video(audioFile)
                (transcription, key) = save_video_to_tmp(audioFile)

        st.session_state["audioFile"] = audioFile

        st.write(key)

        with st.expander("Transcription"):
            st.write(transcription)

        st.session_state["audioTranscription"] = transcription

        summary = summarize(transcription)
        db.update(updates={"summary": summary}, key=key)

        st.write("## Summary")
        st.markdown(summary)

        tasks = tasks(transcription)
        db.update(updates={"tasks": tasks}, key=key)
        st.write("## Tasks")
        st.write(tasks)

        st.success(f"Transcription complete!")


st.subheader("Transcriptions, Summaries and Lists")
db_content = db.fetch().items

for item in db_content:
    with st.expander(item["media"]):
        st.subheader(f'File:  {item["media"]}')
        st.write("##### Summary")
        st.write(item["summary"])
        st.write(item["tasks"])
