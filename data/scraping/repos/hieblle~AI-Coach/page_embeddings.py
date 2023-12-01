import pandas as pd
import numpy as np
import streamlit as st
import openai
import os
import re
import matplotlib.pyplot as plt
import altair as alt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI



# define openai api key and embedding model
openai.api_key = os.getenv('OPENAI_API_KEY')
API_0 = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# create OpenAI instance
llm = OpenAI(
    temperature=0,
    openai_api_key=API_0,
    model_name = GPT_MODEL
)

#------------------#
# 0. streamlit settings/page configuration
st.set_page_config(page_title="Ask your Journal!", page_icon=":sparkles:", layout="wide")
st.sidebar.markdown("# Work with Embeddings")
st.title("Text Embeddings Demo")

st.sidebar.header("Instuctions")
st.sidebar.info("1. Upload your journal entries (if yours doesn't exist already) \n 2. Go to the 'choose' \n 3. Get the most similar text parts")
#------------------#


# create embeddings once (from journal-short.xlsx)
def create_embeddings_from_local():
    df = pd.read_excel("journal-short.xlsx")
    df['embedding'] = df['journal-short'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv('Journal_embedding.csv')

def create_embeddings_from_upload(df):
    df['embedding'] = df['journal-short'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv('pages/Journal_embedding.csv')

def create_embeddings_from_txt_upload(df, filename):
    df['embedding'] = df['Text'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv(f'embeddings/{filename}')
    # todo: check if filename is correct, else use default


def split_text(content):
    """Split content into entries based on date pattern and return a Pandas DataFrame.
       content: string with journal entries
       df: Pandas DataFrame with dates and entries"""
    # Define a regular expression pattern to match dates
    date_pattern = r'\d{4}.\d{2}.\d{2}'

    # Split the content based on the date pattern
    entries = re.split(date_pattern, content)

    # Extract dates from the content
    dates = re.findall(date_pattern, content)

    # Create a dictionary with dates and corresponding entries
    data = {'Date': dates, 'Text': [entry.strip() for entry in entries[1:]]}

    # Create a Pandas DataFrame from the dictionary and return it
    return pd.DataFrame(data)


def langchain_textsplitter(content_entry):
    """Split entries into chunks.
       content_entry: string with one journal entry
       text_array: list of strings with chunks of the entry"""
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    )
    text_array = []
    texts = text_splitter.create_documents([content_entry])
    for chunk in texts:
        text_array.append(chunk.page_content)

    return text_array


# function to call searchbar, userinput
def call_search(filename):
    return filename


# function to search in embeddings
def search(df, search_term, n=3):
    """ 
    df: dataframe with embeddings
    search_term: string to search for
    n: number of results to return
    """
    # convert embeddings to numpy array
    df["embedding"] = df["embedding"].apply(eval).apply(np.array)

    # get embedding of search term
    search_embeddings = get_embedding(search_term, engine="text-embedding-ada-002")

    # calculate cosine similarity
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, search_embeddings))

    # sort by similarity and return top n
    return df.sort_values("similarity", ascending=False).head(n)


def gpt_response(search_term, most_similar):
    """
    gpt_response: generate response to search term based on most similar entries
    search_term: string of question
    most_similar: dataframe with most similar entries
    """
    # concatenate df["Text"] to one string
    conc_text = '\n\n'.join(most_similar["Text"].tolist())
    prompt = f"""You are in the position of a therapist reading over journal entries.
                The patient asks you a question. To answer this question use the information from parts of the journal.
                Question: {search_term}
                Important parts of journal: {conc_text}

                Goals:
                1. Help people to overcome the things which are holding them back.
                2. Discover recurring patterns.
                Answer:

                
                Example
                Question: Why was I feeling so good?
                Important parts of journal: I've meditated a lot. Read a lot.
                Answer: Because meditation and reading were core habits in your life.
    """
    # generate response from gpt
    st.write(llm(prompt))
    # output most similar entries as reference
    st.write("References:")
    for index, row in most_similar.iterrows():
        st.write(row["similarity"], row["Date"], f' "{row["Text"]}"')



def plot_embeddings(df):
    """
    function gets embeddings from dataframe and plots them with TSNE and PCA for downprojection
    df: dataframe with embeddings
    """
    matrix = np.array(df.embedding.apply(eval).to_list())
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    dimTSNE = tsne.fit_transform(matrix)
    x = [x for x,y in dimTSNE]
    y = [y for x,y in dimTSNE]

    st.write("TSNE: ")
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x,y)
    st.pyplot(fig)

    st.write("PCA: ")
    pca = PCA(n_components=2)
    dimPCA = pca.fit_transform(matrix)
    a = [a for a,b in dimPCA]
    b = [b for a,b in dimPCA]

    fig, ax = plt.subplots()
    ax.scatter(a,b)
    st.pyplot(fig)


def plot_embeddings_altair(df):
    """
    downprojection of embeddings with PCA and plot with Altair
    df: dataframe with embeddings
    """
    matrix = np.array(df.embedding.apply(eval).to_list())
    pca = PCA(n_components=2)
    dimPCA = pca.fit_transform(matrix)
    df_PCA = pd.DataFrame(dimPCA, columns=['a', 'b'])
    # include a 3rd feature for color, ex. sentiment
    c = alt.Chart(df_PCA).mark_circle().encode(
        x='a', y='b', tooltip=['a', 'b']).interactive()
    
    st.altair_chart(c, use_container_width=True)




if __name__ == "__main__":

    st.write("Welcome to your personal journal. Here you can search your journal entries and get answers to your questions.")
    st.write("If you want to upload your journal entries, click the tab 'Create new Embedding' below.")
    st.write("If you want to use an existing embedding, choose the 'Use from existing files' tab and click the correct one in the selectbox.")
    st.divider()

    # crawl embeddings folder for existing files
    path_embeddings = "embeddings"
    dir_list = os.listdir(path_embeddings)
    filename = "test.csv"
    tab1, tab2, tab3, tab4 = st.tabs(["1 Create new embeddings", "2 Use from existing files", "3 Search", "4 Plot embeddings"])

    # tab1: create new embeddings
    with tab1:
        st.header("Here you can upload your journal and create a new embedding.")
        st.write("If you've already created your embeddings, jump to the next tab")
        filename = st.text_input(label="First step: Enter a name for your file to save it for later", placeholder="Example: Journal_Leon.csv")
        #if st.button(label="Save", type="primary"):
        #st.write("2. And now upload your existing journal:")
        uploaded_file = st.file_uploader("Choose a .txt-file", type="txt")
        if uploaded_file is not None:
            file = uploaded_file.read().decode("utf-8")
            # split file into entrys, returns pd.Dataframe
            df = split_text(file)
            chunked_df = pd.DataFrame(columns=['Date', 'Text'])
            # iterate over df (entries), chunked_df: new pd.Dataframe filled with chunks
            for index, row in df.iterrows():
                # split text into chunks, return: list of strings
                chunks = langchain_textsplitter(row['Text'])
                date_vector = [row['Date']]*len(chunks)
                # concatenate in new dataframe
                chunked_df = pd.concat([chunked_df, pd.DataFrame({'Date': date_vector, 'Text': chunks})], ignore_index = True)
            # create embeddings from chunked_df
            #filename = "embeddings.csv"
            st.write(f'Folder to save file in is: "embeddings/{filename}"')
            create_embeddings_from_txt_upload(chunked_df, filename)
            st.success("Your embeddings have been created. You can search them now.")
            st.write(chunked_df)

    # tab2: choose embeddings file
    with tab2:
        st.header("Here you can use an existing embedding.")  
        filename = st.selectbox(label= 'List of Embeddings', options=dir_list)
        if st.button(label="Load file", type="primary"):
            data = pd.read_csv(f'{path_embeddings}/{filename}')
            st.success("Your embeddings have been loaded. You can search them now.")
            st.write(data)

    # tab3: search
    with tab3:
        st.write(f'Your chosen journal is: {filename}')
        # now search embeddings (user defines search term, read embeddings from csv, search, return results)
        search_term = st.text_input(label="Search term", placeholder="Enter search term here")
        search_button = st.button(label="Search", type="primary")
        data = pd.read_csv(f'{path_embeddings}/{filename}')
        if search_button:
            # get back most similar terms, used as input for gpt3 (function call to generate response)
            gpt_response(search_term, search(data, search_term))

    # tab4: plot embeddings
    with tab4:
        if st.button('Plot Embeddings with Matplotlib'):
            plot_embeddings(data)
        if st.button('Plot Embeddings with Altair'):
            plot_embeddings_altair(data)






#todo:

# let user choose the name of the embedding file to save it for later
# then choose in the sidebar an embeddings-file or upload a new one


# pinecone?
# - button for restoring saved embeddings (look up in pinecone or local)

# statevariables: für Benutzeroberfläche, Schleifen verschachteln, Namen speichern, Container basteln (expander,...)
