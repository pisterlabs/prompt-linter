import streamlit as st
from htmlTemplates import css, bot_template, user_template

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import whisper,os
from pytube import YouTube
import datetime

#st.set_page_config(page_title="Question and Answering on Web Page", page_icon=":video")



def get_video_text(url):
    model = whisper.load_model('base')
    youtube_video_url = url #"https://www.youtube.com/watch?v=27S2JzQnZh8" #"https://www.youtube.com/watch?v=SuIGHtDyuP8"
    youtube_video = YouTube(youtube_video_url)
    print('youtube_video.title-',youtube_video.title)
    for stream in youtube_video.streams:
        print(stream)
    streams = youtube_video.streams.filter(only_audio=True)
    stream.download(filename='feds.mp4')
    # save a timestamp before transcription
    t1 = datetime.datetime.now()
    # print(f"started at {t1}")
    path=os.getcwd()
    # print(os.getcwd())
    file='feds.mp4'
    # print(os.path.join(path,file))
    full_file_location=os.path.join(path,file)
    # print("full_file_location-",full_file_location)
    # do the transcription
    # output = model.transcribe("fed.mp4", fp16=False)

    output = model.transcribe(full_file_location, fp16=False)
    
# show time elapsed after transcription is complete.
    t2 = datetime.datetime.now()
    # print(f"ended at {t2}")
    # print(f"time elapsed: {t2 - t1}")
    text=output['text']
    # print(text)
    return text
    
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
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


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
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Video", page_icon=":clapper:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with Video :clapper:")
    user_question = st.text_input("Ask a question about your Video:")
    if user_question:
        handle_userinput(user_question)
        
    with st.sidebar:

        st.subheader("Your Video Link")
        url = st.text_input("Enter a Video URL:")

        if st.button("Process"):
            if url:
                with st.spinner("Processing"):
                    if not url.startswith("http"):
                        st.error("Invalid video URL")
                    try:
                        # get pdf text
                        raw_text = get_video_text(url)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Update the processing status in the sidebar
                        st.sidebar.info("Processing completed.")

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore) 
                    except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.error("Please enter video URL.")

if __name__ == '__main__':
    main()