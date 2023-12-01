import os
import time
import pickle
import streamlit as st
from datetime import datetime
from streamlit_chat import message

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma 
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from wiki_content import get_wiki


global embeddings_flag
embeddings_flag = False

st.markdown("<h1 style='text-align: center; color: Red;'>Chat-Wiki</h1>", unsafe_allow_html=True)

buff, col, buff2 = st.columns([1,3,1])
openai_key = col.text_input('OpenAI Key:')
os.environ["OPENAI_API_KEY"] = openai_key

if len(openai_key):

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_key)

    if 'all_messages' not in st.session_state:
        st.session_state.all_messages = []

    def build_index(wiki_content):
        print("building index .....")
        text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,  
            )  
        texts = text_splitter.split_text(wiki_content)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        with open("./embeddings.pkl", 'wb') as f:
            pickle.dump(docsearch, f)

        return embeddings, docsearch

    # Create a function to get bot response
    def get_bot_response(user_query, faiss_index):
        docs = faiss_index.similarity_search(user_query, K = 6)
        main_content = user_query + "\n\n"
        for doc in docs:
            main_content += doc.page_content + "\n\n"
        messages.append(HumanMessage(content=main_content))
        ai_response = chat(messages).content
        messages.pop()
        messages.append(HumanMessage(content=user_query))
        messages.append(AIMessage(content=ai_response))

        return ai_response

    # Create a function to display messages
    def display_messages(all_messages):
        for msg in all_messages:
            if msg['user'] == 'user':
                message(f"You ({msg['time']}): {msg['text']}", is_user=True, key=int(time.time_ns()))
            else:
                message(f"IA-Bot ({msg['time']}): {msg['text']}", key=int(time.time_ns()))

    # Create a function to send messages
    def send_message(user_query, faiss_index, all_messages):
        if user_query:
            all_messages.append({'user': 'user', 'time': datetime.now().strftime("%H:%M"), 'text': user_query})
            bot_response = get_bot_response(user_query, faiss_index)
            all_messages.append({'user': 'bot', 'time': datetime.now().strftime("%H:%M"), 'text': bot_response})

            st.session_state.all_messages = all_messages
            
        
    # Create a list to store messages

    messages = [
            SystemMessage(
                content="You are a Q&A bot and you will answer all the questions that the user has. If you dont know the answer, output 'Sorry, I dont know' .")
        ]

    search = st.text_input("What's on your mind?")

    if len(search):
        wiki_content, summary = get_wiki(search)

        if len(wiki_content):
            try:
                # Create input text box for user to send messages
                st.write(summary)
                user_query = st.text_input("You: ","", key= "input")
                send_button = st.button("Send")

                if len(user_query) and send_button:
                    # Create a button to send messages
                    if not embeddings_flag:
                        embeddings, docsearch = build_index(wiki_content)
                        embeddings_flag = True
                        with open("./embeddings.pkl", 'rb') as f: 
                            faiss_index = pickle.load(f)
                # Send message when button is clicked
                if embeddings_flag:
                    send_message(user_query, faiss_index, st.session_state.all_messages)
                    display_messages(st.session_state.all_messages)

            except:
                st.write("something's Wrong... please try again")
            

