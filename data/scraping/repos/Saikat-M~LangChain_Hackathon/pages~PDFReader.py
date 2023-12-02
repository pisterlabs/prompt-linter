import os 
import base64
import numpy as np

from apikeys import OPENAI_API_KEY, HUGGINGFACE_API_KEY

import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.llms import OpenAI

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub

from google.cloud import texttospeech_v1

from audio_recorder_streamlit import audio_recorder


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

st.title('PDF Q&A')
st.divider()

llm = OpenAI(temperature=0)
text_splitter = CharacterTextSplitter()

uploaded_file = st.file_uploader("Upload a file")
q_and_a_prompt = st.text_input('Please type in your question', key='question')


def text_to_speech(text):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"./texttospeechconversion-jsonkey.json"

  print("text", text)

  # Instantiates a client
  client = texttospeech_v1.TextToSpeechClient()

  # Set the text input to be synthesized
  synthesis_input = texttospeech_v1.SynthesisInput(text=text)


  voice = texttospeech_v1.VoiceSelectionParams(
      language_code='en-US', ssml_gender=texttospeech_v1.SsmlVoiceGender.FEMALE
  )

  # Select the type of audio file you want returned
  audio_config = texttospeech_v1.AudioConfig(
      # https://cloud.google.com/text-to-speech/docs/reference/rpc/google.cloud.texttospeech.v1#audioencoding
      audio_encoding=texttospeech_v1.AudioEncoding.MP3
  )

  # Perform the text-to-speech request on the text input with the selected
  # voice parameters and audio file type
  response = client.synthesize_speech(
      input=synthesis_input, voice=voice, audio_config=audio_config
  )

  # The response's audio_content is binary.
  return response


if uploaded_file:
    # Create the directory if it doesn't exist
    os.makedirs('files', exist_ok=True)
    
    # Save the uploaded file to the specified directory
    with open(os.path.join('files', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Creating the file path to load the document
    loader = PyPDFLoader("./files/"+uploaded_file.name)
    pages = loader.load_and_split()
    # st.write(pages[0].page_content)

    ##Wrting Summary
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(pages)
    with st.container():
        st.text_area("Summary", value = summary, disabled=True, height=190)
        audio_bytes = text_to_speech(summary)

        st.audio(audio_bytes.audio_content, format='audio/ogg')
    ##Writing Summary

    ##Q&A
    #Creating Embedding and VectorDB out of it
    embeddings = OpenAIEmbeddings()
    vectorDb = Chroma.from_documents(pages, embedding=embeddings)

    #Creating ChatVector using OpenAI model
    pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"), 
                                        vectorDb, 
                                        return_source_documents = True,
                                        # memory=memory
                                        )
    # st.write("filename:", uploaded_file.name)

    if q_and_a_prompt: 
        #Sending the Query to ChatVector to get answers
        result = pdf_qa({"question": q_and_a_prompt, "chat_history": ''})
        st.write(result["answer"])
        # st.write(translate_text(result["answer"], 'Bengali'))
        audio_bytes = text_to_speech(result["answer"])
        st.audio(audio_bytes.audio_content, format='audio/ogg')
    ##Q&A

