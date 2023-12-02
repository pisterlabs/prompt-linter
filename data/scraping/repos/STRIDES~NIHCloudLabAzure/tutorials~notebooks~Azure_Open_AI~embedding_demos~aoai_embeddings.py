import openai
from openai.embeddings_utils import get_embedding, cosine_similarity # must pip install openai[embeddings]
import pandas as pd
import numpy as np
import os
import streamlit as st
import time
from PIL import Image
from dotenv import load_dotenv

# load in .env variables
load_dotenv()

# configure azure openai keys
openai.api_type = 'azure'
openai.api_version = os.environ['AZURE_OPENAI_VERSION']
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_key = os.environ['AZURE_OPENAI_KEY']

def embedding_create():
    # acquire the filename to be embed
    st.subheader("Vector Creation")
    st.write('This program is designed to embed your pre-chunked .csv file. \
                By accomplishing this task, you will be able to chat over all cotent in your .csv via vector searching. \
                    Just enter the file and the program will take care of the rest (specify file path if not in this directory). \
                        Welcome to CHATGPT over your own data !!')
    filename = st.text_input("Enter a file: ", key='filename', value="")

    # start the embeddings process if filename provided
    if filename:
        
        # read the data file to be embed 
        df = pd.read_csv('C:\\src\\AzureOpenAI_Gov_Workshop\\' + filename)
        st.write(df)

        # calculate word embeddings 
        df['embedding'] = df['text'].apply(lambda x:get_embedding(x, engine='text-embedding-ada-002'))
        df.to_csv('C:\\src\\AzureOpenAI_Gov_Workshop\\microsoft-earnings_embeddings.csv')
        time.sleep(3)
        st.subheader("Post Embedding")
        st.success('Embeddings Created Sucessfully!!')
        st.write(df)


def embeddings_search():

    # Streamlit configuration
    st.subheader("Vector Search")
    st.write('This program is designed to chat over your vector stored (embedding) .csv file. \
                This Chat Bot works alongside the "Embeddings Bot" Chat Bot. \
                    Be specific with the information you want to obtain over your data. \
                        Welcome to CHATGPT over your own data !!')
    if 'answer' not in st.session_state:
        st.session_state.answer = []  
    if 'score' not in st.session_state:
        st.session_state.score = []     
    if 'past' not in st.session_state:
        st.session_state.past = []  

    # read in the embeddings .csv 
    # convert elements in 'embedding' column back to numpy array
    df = pd.read_csv('C:\\src\\AzureOpenAI_Gov_Workshop\\microsoft-earnings_embeddings.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    # caluculate user query embedding 
    search_term = st.text_input("Enter a search query: ", key='search_term', placeholder="")
    if search_term:
        st.session_state.past.append(search_term)
        search_term_vector = get_embedding(search_term, engine='text-embedding-ada-002')

        # find similiarity between query and vectors 
        df['similarities'] = df['embedding'].apply(lambda x:cosine_similarity(x, search_term_vector))
        df1 = df.sort_values("similarities", ascending=False).head(5)

        # output the response 
        answer = df1['text'].loc[df1.index[0]]
        score = df1['similarities'].loc[df1.index[0]]
        st.session_state.answer.append(answer)
        st.session_state.score.append(score)
        with st.expander('Vector Search'):
            for i in range(len(st.session_state.answer)-1, -1, -1):
                st.info(st.session_state.past[i])
                st.write(st.session_state.answer[i])
                st.write('Score: ', st.session_state.score[i])


def main():
    # Streamlit config
    st.title("Demo-Azure OpenAI Embeddings")
    image = Image.open('image_logo2.png')
    st.image(image, caption = '')
    st.sidebar.title('Chat Bot Type Selection')
    chat_style = st.sidebar.selectbox(
        'Choose between Embeddings Bot or Search Bot', ['Embeddings Bot','Search Bot']
    )
    if chat_style == 'Embeddings Bot':
        embedding_create()
    elif chat_style == 'Search Bot':
        embeddings_search()

if __name__ == '__main__':
    main()