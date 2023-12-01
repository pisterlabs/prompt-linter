from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from openai_key import api_key
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
import os
import urllib.request
import json
import urllib
import streamlit as st
from streamlit_chat import message



def set_openai_key(overwrite=False):
    if not os.environ.get("OPENAI_API_KEY") or overwrite:
        print("Resetting OPENAI KEY")
        os.environ["OPENAI_API_KEY"]=api_key
    else:
        print("Reusing OPENAI KEY from Env")





def create_db_and_title_from_url(url, chunk_size, chunk_overlap):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript= loader.load()
    title = get_video_title(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap)
    docs= text_splitter.split_documents(transcript)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs,embeddings)
    return db, title

def get_video_title(url):
    params = {"format": "json", "url": url}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
    return(data['title'])


def get_chain(temp, db):
    llm = OpenAI(temperature=temp)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="map_reduce")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain(
        retriever=db.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory
    )
    return chain

def reset():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    if 'input' in st.session_state and st.session_state['input']:
        st.session_state['input']="Hello, Let's start?"
    

def get_text():
    input_text = st.text_input("You","Hello, Let's start?", key="input")
    return input_text

# keep the bot running in a loop to simulate a conversation
def main():
    st.set_page_config(
        page_title="🧞🧞 Bismayan's Youtube Genie",
        page_icon="🧞🧞",
        layout="wide",
        initial_sidebar_state="expanded")
    col1, col2 = st.columns([1,3])
    with col1:
        st.image("./logo.png", width=200)
    with col2:
        st.markdown("<h1 style='text-align: left; color: Black;'> Welcome to Bismayan's Youtube Genie </h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: Black;'>Your AI agent for answering questions about Youtube Videos </h3>", unsafe_allow_html=True)
        
    with st.sidebar:
        st.title("Advanced Config")
        temp= st.slider("Creativity", min_value=0.1, max_value=0.95, step=0.05, value=0.2)
        chunk_size= st.slider("Size of Chunks", min_value=200, max_value=1500, step=100, value=1000)  
        chunk_overlap= st.slider("Overlap between Chunks", min_value=0, max_value=500, step=50, value=0)    

    st.empty()
    
    col1, _ = st.columns([3,1])
    with col1:    
        url = st.text_input(
            "Enter a Youtube Video URL. Wordy Videos please and no shorts!", value="", on_change=reset)
        if url:
            db,title = create_db_and_title_from_url(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.text(f"The Genie has loaded Information about the video titled: {title}")
            chain=get_chain(temp=temp, db=db)

        # Storing the chat
            if 'generated' not in st.session_state or len(st.session_state['generated'])==0:
                st.session_state['generated'] = []

            if 'past' not in st.session_state or len(st.session_state['past'])==0:
                st.session_state['past'] = []
            user_input = get_text()
            
            reset_button = st.button("Reset Conversation", key="conv_butt")
            if reset_button:
                st.session_state['generated'] = []
                st.session_state['past'] = []
                db,title = create_db_and_title_from_url(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chain=get_chain(temp=temp, db=db)
                user_input="Hello, Let's start?"
            if user_input:
                with st.spinner("Finding that for you..."):
                    if user_input=="Hello, Let's start?":
                        output = "Hello this is Youtube Genie. How may I be of service?"
                    else:
                        output = chain(
                            {"question": user_input})["answer"]
                    # store the output 
                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(output)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    if st.session_state['past'][i]!="Hello, Let's start?":
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    
if __name__ == "__main__":
    st.runtime.legacy_caching.clear_cache()
    set_openai_key()
    main()