# Import the necessary modules
import streamlit as st
import time
import os
import textwrap
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json

# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="FVA", page_icon=":evergreen_tree:")

# Load Pinecone
load_dotenv()

# Variables
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
index_name = "moflibrary"
namespace = "moflibrary"
text_field = 'text'

# Vector Store
@st.cache_resource
def initialize_pinecone_index():
    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index(index_name)
    embedding = OpenAIEmbeddings()
    return embedding, index

# Call the function to initialize Pinecone and get the embedding and index objects
embedding, index = initialize_pinecone_index()

# Create the vector store and other components outside the cached function
vectorstore = Pinecone(index, embedding.embed_query, text_field)
docsearch = Pinecone.from_existing_index(index_name=index_name, namespace=namespace, embedding=embedding)

with open('pages/Library.json', 'r') as f:
    library = json.load(f)

# Convert the list of documents into a dictionary with the IDs as keys
library_dict = {doc['id']: doc for doc in library}

# VA Setup
turbo_llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# Functions to format the response
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Main area
st.title('Forestry Virtual Assistant')

# Sidebar
st.sidebar.title('Welcome')

# How to use section
how_to_use = st.sidebar.expander('How to Use')
with how_to_use:
    st.markdown('<p style="font-size:10px">1. Enter your question related to forestry in the text input field in the main area of the app.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">2. If you want to see the sources that the answer is based on, check the "Show Sources" checkbox.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">3. If you havve checked "Show Sources", you can choose to see specific metadata from the sources. Check the corresponding checkboxes under "Select Metadata".</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">4. Select the AI model you want to use from the dropdown menu. The options are "gpt-3.5-turbo-0613" and "gpt-4-0613".</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">5. Click the "Submit" button to get the answer to your question. The AI will search through a library of forestry documents for the most relevant information and present it as a clear and concise answer.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">6. The answer will appear in the main area of the app. If you checked the "Show Sources" checkbox, the sources will appear below the answer. Each source can be expanded to view its content and metadata.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">7. If you want to ask another question, simply enter it in the text input field and click "Submit" again. The text input field can be cleared by clicking the "x" on the right side of the field.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px">8. The total runtime and the model used for the last query are displayed at the bottom of the main area.</p>', unsafe_allow_html=True)


# About section
about = st.sidebar.expander('About')
with about:
    st.markdown('<p style="font-size:10px">This is the Forestry Virtual Assistant - FVA, a tool designed to provide quick and accurate answers to your forestry-related questions. The FVA uses advanced AI models to search through a comprehensive library of forestry documents and provide responses based on the most relevant information. The library includes a wide range of documents from the BC Ministry of Forests, Timber Pricing Branch, Timber Supply Review, and more. Whether you are looking for specific information or general knowledge, the FVA is here to assist you.</p>', unsafe_allow_html=True)

# FAQ section
faq = st.sidebar.expander('FAQ')
with faq:
    st.markdown('<p style="font-size:10px"><b>Q1: How does the Forestry Virtual Assistant work?</b><br>A1: The FVA uses AI models to understand your question and search through a library of forestry documents for the most relevant information. It then presents this information as a clear and concise answer.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px"><b>Q2: What kind of questions can I ask?</b><br>A2: You can ask any question related to forestry. The FVA is designed to handle a wide range of topics, from specific details about forestry practices to general information about the forestry sector.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px"><b>Q3: How accurate are the answers?</b><br>A3: The FVA strives to provide the most accurate information possible. However, as with any AI tool, the accuracy can vary depending on the complexity of the question and the information available in the library documents.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px"><b>Q4: Can I see the sources of the information?</b><br>A4: Yes, you can choose to see the sources of the information by checking the "Show Sources" checkbox before submitting your question.</p>', unsafe_allow_html=True)

query = st.text_area('Enter Your Question:', height=100)
show_sources = st.checkbox('Show Sources')

if show_sources:
    # Add checkboxes for each metadata category
    st.markdown('<p style="font-size:80%">Select Metadata:</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_id = st.checkbox('Show ID')
        show_author = st.checkbox('Show Author')
    with col2:
        show_subject = st.checkbox('Show Subject')
        show_creator = st.checkbox('Show Creator')
    with col3:
        show_mod_date = st.checkbox('Show Modification Date')
        show_creation_date = st.checkbox('Show Creation Date')
    with col4:
        show_keywords = st.checkbox('Show Keywords')

# Model selection
model_name = st.selectbox('Select model:', ('gpt-3.5-turbo-16k', 'gpt-4'))

if st.button('Submit'):
    start_time = time.time() # Moved the start time here
    llm_response = qa_chain({"query": query})
    response_text = wrap_text_preserve_newlines(llm_response['result'])
    st.write(response_text)

    if show_sources:
        st.write('\n\nSources:')
        for i, source in enumerate(llm_response["source_documents"]):
            title = source.metadata.get("Title", "N/A")  # Always show the title
            with st.expander(f'Source {i+1}: {title}'):
                # Extract the metadata from the source document
                id = source.metadata.get("id", "N/A") if show_id else None
                author = source.metadata.get("Author", "N/A") if show_author else None
                subject = source.metadata.get("Subject", "N/A") if show_subject else None
                creator = source.metadata.get("Creator", "N/A") if show_creator else None
                creation_date = source.metadata.get("CreationDate", "N/A") if show_creation_date else None
                mod_date = source.metadata.get("ModDate", "N/A") if show_mod_date else None
                keywords = source.metadata.get("Keywords", "N/A") if show_keywords else None

                # Print the metadata
                if id: 
                    link = library_dict[id]['Link']
                    hyperlink = f'<a href="{link}" target="_blank">{id}</a>'
                    st.markdown(hyperlink, unsafe_allow_html=True)
                if author: st.write(f"Author: {author}")
                if subject: st.write(f"Subject: {subject}")
                if creator: st.write(f"Creator: {creator}")
                if creation_date: st.write(f"Creation Date: {creation_date}")
                if mod_date: st.write(f"Modification Date: {mod_date}")
                if keywords: st.write(f"Keywords: {keywords}")

                # Print the page content
                page_content = source.page_content
                wrapped_text = wrap_text_preserve_newlines(page_content)
                st.write(f"\nSource Content: {wrapped_text}")

    end_time = time.time()
    total_time = end_time - start_time
    st.write("\nTotal runtime: {:.2f} seconds".format(total_time))
    st.write(f"Model used: {model_name}")




# streamlit run 1_Homepage.py