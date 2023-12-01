import streamlit as st

from chromadb.config import Settings
from dotenv import load_dotenv
import argparse
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#Page config
st.set_page_config(page_title='Amberg Loglay Playground', page_icon='assets/AE logo only.png', layout='wide', initial_sidebar_state='expanded')

#markdown for sidebar main page
st.title("Chat with your Emails")
st.sidebar.title('Models')

selection = st.sidebar.radio("Choose:", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"], label_visibility='hidden' )

#define which model to use based on selection
if selection == "gpt-3.5-turbo":
    model = "gpt-3.5-turbo"
elif selection == "gpt-3.5-turbo-16k":
    model = "gpt-3.5-turbo-16k"
elif selection == "gpt-4":
    model = "gpt-4"


st.sidebar.title('Collection')
selection = st.sidebar.radio("Choose:", ["500 Chunksize", "1000 Chunksize"], label_visibility='hidden' )

#define which collection to use based on selection
if selection == "500 Chunksize":
    collection_name = "openai_ada"
elif selection == "1000 Chunksize":
    collection_name = "openai_ada_1000cs"

#add a description of the model in the subtitle
st.subheader("Model: " + model + " through Chat API")
#add a description of the subheader
st.markdown("This is a playground for the evaluation of a prototype. The OpenAI API is used to answer questions about my emails. The email content is stored in a database and is queried using the Chroma library. The Chroma library is used to retrieve the most relevant documents for a given query. The retrieved emails are then used to answer the question using the API. The answer is then displayed in the output area below. The answer is not stored in the database.")

#add a slider for the temperature
temperature = st.sidebar.slider('Temperature ("creativity of the model")', 0.0, 1.0, 0.0, 0.1)

#add a slider for the max tokens
if model == "gpt-4" or model == "gpt-3.5-turbo":
    max_tokens = st.sidebar.slider('Max Tokens ("length of the answer")', 0, 3000, 1000, 100)
if model == "gpt-3.5-turbo-16k":
    max_tokens = st.sidebar.slider('Max Tokens ("length of the answer")', 0, 14000, 2000,  500)

#add a slider for number of documents
num_docs = st.sidebar.slider('Number of Documents ("number of documents to retrieve")', 0, 10, 3, 1)



#create text area input for query
query_string = st.text_area('Enter a query', 'Wann habe ich mit Oliver Hunziker über das Email Archiv diskutiert?')

#add execute button
execute = st.button('Execute Query')

#add output area for answer
output = st.empty()
output.subheader("Answer:")

#add free space between answer and source documents
st.markdown("")
st.markdown("")



#add output area for source documents (if not hidden)
output_docs = st.empty()
output_docs.subheader("Source Documents:")
#output_docs.markdown("The following documents were retrieved from the database:")



#os.environ["OPENAI_API_KEY"] == st.secrets["openai_key"]



load_dotenv()

# Define the folder for storing database
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY_STREAMLIT')
persist_directory = os.environ.get('PERSIST_DIRECTORY_STREAMLIT')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='EmailGPT: Ask questions to your Emails using the power of OpenAI.')
                                                 
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


def query_chat(query_string, num_docs=3, max_tokens=2000, temperature=0.5, model_name="gpt-3.5-turbo"):

    # Parse the command line arguments
    args = parse_arguments()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(collection_name=collection_name, persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens, callbacks=callbacks, verbose=False, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
    # Get the answer from the chain
    res = qa(query_string)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents'][0:num_docs]
    
    return answer, docs


#extract page content and metadata from a document
def extract_doc(document):
    # Split the string into lines
    lines = document.page_content.split('\n')

    # Initialize an empty list to hold the formatted lines
    formatted_lines = []

    # Iterate through each line
    for line in lines:
        if line.startswith('- '):  # If the line is a list item, keep it as it is
            formatted_lines.append(line)
        
        else:  # If the line is empty, keep it as it is
            formatted_lines.append(line)

    # Join the formatted lines back into a single string, with line breaks between lines
    formatted_string = '\n'.join(formatted_lines)
    source = document.metadata['web_link'] 
    return formatted_string, source


#run query_chat(openai_key, query_string) from query.py with when execute button is clicked
if execute:
    
    answer, docs = query_chat( query_string, num_docs=num_docs,
                                     max_tokens=max_tokens, temperature=temperature, model_name=model)
    output.write(answer)
    for doc in docs:
        content, source = extract_doc(doc)
        st.markdown("Source:  \n" + source + "  \n  \nContent:  \n" + content)
        st.divider()

