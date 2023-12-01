from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,NLTKTextSplitter,SpacyTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains import LLMChain,QAGenerationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import streamlit as st
import os
import random
import itertools
from io import StringIO
import PyPDF2

import whisper
from pytube import YouTube


import sys
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

ss = st.session_state

def youtube_whisper(video_url):

    audio_file = YouTube(video_url).streams.filter(only_audio=True).first().download(filename="./video/audio.mp4")

    whisper_model = whisper.load_model("base")

    transcription = whisper_model.transcribe("./video/audio.mp4")
        
    return transcription['text']


def create_db_from_youtube_video_url(video_url,chunks,overlap):
    
    loader = YoutubeLoader.from_youtube_url(video_url)

    docs = loader.load()
    
    #transcript_w = youtube_whisper(video_url)
    #whisper_trans = [Document(page_content=t) for t in [str(transcript_w)]]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
    #docs = text_splitter.split_documents(whisper_trans)
    docs_ = text_splitter.split_documents(docs)
    
    db = FAISS.from_documents(docs_, OpenAIEmbeddings())

    return db,docs


def get_response_from_query(db, query, model_name,temp, summary = False):
   
    docs = db.similarity_search(query, k=5)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name=model_name, temperature=temp)

    if summary: 
        template = """
        You are a helpful assistant that that can summarize youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript.
        
        Be aware of the ads and other non-relevant information. If encountered, skip them.

        Your answers should as clear and detailed as possible.
        """
    else:
        template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be clear. If the question is too broad, ask for clarification.
        """


    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    #response = response.replace("\n", "")
    return response, docs_page_content


def process_pdf(text,chunks,user_question,temp,open_ai_key):

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        chain = load_qa_chain(OpenAI(temperature=temp), chain_type="map_reduce")
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return response["output_text"]
        


def create_db_from_pdf(loaded_docs,chunk_size,overlap):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # number of characters in each chunk
        chunk_overlap=overlap, # number of overlapping characters for each chunk 
    )
    
    text_split = text_splitter.split_text(loaded_docs)

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(text_split, embeddings)

    return knowledge_base,text_split

def new_process_pdf(knowledge_base,user_question,model_name,temp,summary = False):
        
    docs = knowledge_base.similarity_search(user_question, k=4)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name=model_name, temperature=temp)

    # Template to use for the system message prompt
    if summary: 
        template = """
        You are a helpful assistant that can summarize a document using only factual information from it: {docs}
        
        Your summary should as be as clear and detailed as possible.

        """
    else:
        template = """
        You are a helpful assistant that can answer questions about pdf and text files: {docs}
        
        Only use the factual information from files to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be concise and clear.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=user_question, docs=docs_page_content)
    return response
    
def load_docs(files):
    
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)

            information = pdf_reader.metadata


            txt = f"""
                Author: {information.author}
                Creator: {information.creator}
                Producer: {information.producer}
                Subject: {information.subject}
                Title: {information.title}
                """

            #print(txt)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text



def split_texts(text,chunk_size,overlap):

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, # number of characters in each chunk
            chunk_overlap=overlap, # number of overlapping characters for each chunk 
        )
    text_split = text_splitter.split_text(text)

    if not text_split:
        st.error("Document failed to split")

    return text_split


def split_text_nonrecursive(text,chunk_size,overlap):

    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = chunk_size,
        chunk_overlap  = overlap,
        length_function = len)

    text_split = text_splitter.split_text(text)

    return text_split


def generate_random_questions(text, n_questions,model_name,temp,chunk):

    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(n_questions)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]


    chain = QAGenerationChain.from_llm(ChatOpenAI(model_name=model_name, temperature=temp))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            #st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    print(eval_set_full)
    return eval_set_full
