import os
import streamlit as st
import pinecone
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from html2docx import html2docx
import markdown
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pdfkit

# Ovaj kod ce biti koriscen kao deo streamlit aplikacije koja ce pisati dokumente u stilu neke osobe
# jedna od upotreba ce biti u Multi Tool Botu kao jedan od toolova za agenta


def main():
    # Retrieving API keys from env
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    # Initialize Pinecone
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
                  environment=os.environ["PINECONE_API_ENV"])
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    # Define metadata fields
    metadata_field_info = [
        AttributeInfo(name="person_name",
                      description="The name of the person", type="string"),
        AttributeInfo(
            name="topic", description="The topic of the document", type="string"),
        AttributeInfo(
            name="text", description="The Content of the document", type="string"),
    ]
    # Define document content description
    document_content_description = "Content of the document"
    # Initialize OpenAI embeddings and LLM
    # markdown to html

    with st.sidebar:
        model = st.selectbox(
            "Odaberite model",
            ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4")
        )
        temp = st.slider(
            'Set temperature (0=strict, 1=creative)', 0.0, 1.0, step=0.1)
        # Initializing ChatOpenAI model
    llm = ChatOpenAI(model_name=model, temperature=temp,
                     openai_api_key=openai_api_key)

    # Create VectorStore from index
    namespace = "stilovi"
    index_name = "embedings1"

    # Streamlit app
    st.subheader('Write in the style of indexed people')
    st.caption("App omogucava da se pronadje tekst odredjene osobe na odredjenu temu i da se koristi kao osnova za pisanje teksta u stilu te osobe")
    st.caption(
        "Kad bude dovoljno usera sa svojim stilovima, stil se moze odrediti na osnovu imena prijavljenog usera")
    with st.sidebar:
        st.caption("Ver 20.07.23")

    with st.form(key='stilovi', clear_on_submit=True):
        query = st.selectbox(
            "Choose person name to use their style:", ("", "Sean Carroll", "Dragan Varagic", "Neuka osoba"))
        tema = st.text_input(
            'Enter the instructions like lenght, topic, language...:')
        submit_button = st.form_submit_button(label='Submit')
    text = "text"
    osnova = ""
    odgovor = "Odgovor"
    if submit_button:
        # Perform self-query retrieval with retrieval limit
        filter = {'person_name': {"$eq": query}}
        # retriever= retriever.get_relevant_documents(query, search_type="mmr", search_kwargs={ "filter" : filter}, search_kwargs={"k": 3})
        with st.spinner("Trazim stil..."):
            vectorstore = Pinecone.from_existing_index(
                index_name, embeddings, text, namespace=namespace)
            retriever = SelfQueryRetriever.from_llm(
                llm, vectorstore, document_content_description, metadata_field_info, enable_limit=True, verbose=True, search_kwargs={"k": 3, "filter": filter})
            results = retriever.get_relevant_documents(query)

            # Display results
            for result in results:
                osnova += result.page_content

            prompt = f"Kao vrhunski bloger napisi {tema} koristeci ovaj primer kao osnovu za stil pisanja: {osnova}. Definisi naslov i podnaslove. Koristi markdown za naslov i podnaslove kao H1, H2 itd."
        with st.spinner("Pisem tekst..."):
            odgovor = llm.predict(prompt)
            st.markdown(odgovor)
    html = markdown.markdown(odgovor)
    # # html to docx
    buf = html2docx(html, title="Zapisnik")
    # create pdf
    options = {
        'encoding': 'UTF-8',  # Set the encoding to UTF-8
        'no-outline': None,
        'quiet': ''
    }

    pdf_data = pdfkit.from_string(html, False, options=options)

    with st.sidebar:
        st.download_button("Download as .txt",
                           odgovor, file_name="TekstuStilu.txt")
        st.download_button(label="Download as .pdf",
                           data=pdf_data,
                           file_name="TekstuStilu.pdf",
                           mime='application/octet-stream')
        st.download_button(
            label="Download .docx",
            data=buf.getvalue(),
            file_name="TekstuStilu.docx",
            mime="docx"
        )


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(
    'Login to text summarizer', 'main')

if st.session_state["authentication_status"]:
    with st.sidebar:
        authenticator.logout('Logout', 'main', key='unique_key')
    # if login success run the program
    main()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


