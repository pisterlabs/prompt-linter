###################################################################################################
#  Brian Lesko               2023-11-14
#  This code implements an embedding shema that is used to compare the similarity of textual data.
#  Think of it as an upgraded Cmd+F search. Written in pure Python & created for learning purposes.
###################################################################################################

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from customize_gui import gui
from api_key import openai_api_key
from openai import OpenAI
import tiktoken as tk 
api_key = openai_api_key
client = OpenAI(api_key = api_key)
gui = gui()
import plotly.express as px

def get_embedding(text, model="text-embedding-ada-002",encoding_format="float"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input = text, model=model)
    #st.write(response.data[0].embedding) # Debug : why the hell did OpenAI structure it like this? 
    return response.data[0].embedding

@st.cache_resource
def chunk_tokens(tokens, chunk_length=40, chunk_overlap=10):
    chunks = []
    for i in range(0, len(tokens), chunk_length - chunk_overlap):
        chunks.append(tokens[i:i + chunk_length])
    return chunks

@st.cache_resource
def detokenize(tokens):
    enc = tk.encoding_for_model("gpt-4")
    text = enc.decode(tokens)
    return text

@st.cache_resource
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
    return embeddings

def get_text(upload):
    # if the upload is a .txt file
    if upload.name.endswith(".txt"):
        document = upload.read().decode("utf-8")
    return document

def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}" + """ In your answer please be clear and concise, sometime funny.
        If you need to make an assumption you must say so."""
    )
    return augmented_query

class document:
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.tokens = self.tokenize()
        self.token_chunks = chunk_tokens(self.tokens, chunk_length=50, chunk_overlap=10)
        self.text_chunks = [detokenize(chunk) for chunk in self.token_chunks]
        self.chunk_embeddings = embed_chunks(self.text_chunks)
        self.embedding = get_embedding(self.text)
        self.df = pd.DataFrame({
            "name": [self.name], 
            "text": [self.text], 
            "embedding": [self.embedding], 
            "tokens": [self.tokens], 
            "token_chunks": [self.token_chunks], 
            "text_chunks": [self.text_chunks], 
            "chunk_embeddings": [self.chunk_embeddings]
            })

    def similarity_search(self, query, n=3):
        query_embedding = get_embedding(query)
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            similarities.append(cosine_similarity([query_embedding], [chunk_embedding])[0][0])
        # the indicies of the top n most similar chunks
        idx_sorted_scores = np.argsort(similarities)[::-1]
        context = ""
        for idx in idx_sorted_scores[:n]:
            context += self.text_chunks[idx] + "\n"
        return context
    
    def similarity(self, doc):
        return cosine_similarity([self.embedding], [doc.embedding])[0][0]
    
    def tokenize(self):
        enc = tk.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(self.text)
        return tokens

def main():
    gui.clean_format()
    with st.sidebar:
        gui.about(text="This code implements text embedding, check it out!")
    gui.display_existing_messages(intro = "Hi, I'm going to help you understand what an embedding is - and why it's useful. Let's get started by entering some text to embed.")
    text = st.chat_input("Write a message")
    if text:
        if "texts" not in st.session_state:
                st.session_state.texts = []
        st.session_state.texts.append(text)
        doc = document("User Input", text) # document class defined above
        with st.sidebar:
            st.markdown("""---""")
            st.subheader("Your text:")
            st.write(doc.text)
            st.write("Model used: text-embedding-ada-002")
        with st.chat_message("assistant"):
            if "soccer_similarities" not in st.session_state:
                st.session_state.soccer_similarities = []
            if "math_similarities" not in st.session_state:
                st.session_state.math_similarities = []
            # Example of similarity
            soccer = document("Soccer", "Eleven players per team. Kicking a ball with your feet. Attempting to score a goal against the other team. The game is 90 minutes long. This is soccer. ")
            math = document("Math", "Using numbers doing calculations with a variety of variables to solve equations. Algebra, calculus, differential equations, geometry, and trigonometry are all types of math. How to solve problems. This is math.")
            soccer_similarity = doc.similarity(soccer)  
            math_similarity = doc.similarity(math)
            st.session_state.soccer_similarities.append(soccer_similarity)
            st.session_state.math_similarities.append(math_similarity)  
            st.markdown(f"""
                Here's an example of how an embedding can be used:
                """)
            df = pd.DataFrame({
                'soccer_similarities': st.session_state.soccer_similarities,
                'math_similarities': st.session_state.math_similarities,
                'texts': st.session_state.texts
            })
            fig = px.scatter(df, x='soccer_similarities', y='math_similarities', hover_data=['texts'])
            fig.update_traces(hovertemplate='Text: %{customdata[0]}<extra></extra>')
            fig.update_layout(
                title="Similarity to soccer vs. Similarity to math",
                xaxis_title="Similarity to soccer",
                yaxis_title="Similarity to math",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
            st.plotly_chart(fig)
        


        with st.chat_message("assistant"):
                st.write("The embedded text looks like this: ")
                st.dataframe(np.array(doc.embedding).reshape(1, -1))
        with st.chat_message("assistant"):
                st.markdown(f"""
                    Here's why it's useful
                    - Searching through documents.
                    - Comparing the similarity of two pieces of text.
                    - Text generation - think ChatGPT.
                    """) 
        with st.chat_message("assistant"):
                length = np.array(doc.embedding).shape
                st.markdown(f"""
                    Here's how it works
                    - Each time text is embedded, the result is a vector of values that 'describe' the text.
                    - No matter the input text, the output is always the same length: {length}
                    - For this reason, text is often chunked into smaller pieces, and each piece is embedded.
                    - Similarity between two embeddings is calculated using L2 distance or cosine similarity. 
                    """)
main()