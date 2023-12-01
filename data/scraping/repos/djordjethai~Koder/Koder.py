# This code is used to create a code based on LangChain Library using streamlit for web interface

# Import necessary libraries
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone
import streamlit as st
import os
from myfunc.mojafunkcija import st_style, positive_login, init_cond_llm, show_logo


# these are the environment variables that need to be set for LangSmith to work
os.environ["LANGCHAIN_PROJECT"] = "Koder"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ.get("LANGCHAIN_API_KEY")

version = "05.11.23. (Streamlit, Pinecone, LangChain)"

st.set_page_config(page_title="Koder", page_icon="🖥️", layout="wide")
st_style()


def main():
    # Set text field
    text_field = "text"

    # Insert path to PythonGPT3Tutorial

    # Read API keys from env
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # Retrieving API keys from env
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

    # Initialize OpenAIEmbeddings and Pinecone
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Create Pinecone index
    index = pinecone.Index("embedings1")
    name_space = "koder"
    vectorstore = Pinecone(index, embeddings, text_field, name_space)
    show_logo()
    # Get user input
    st.markdown(f"<p style='font-size: 10px; color: grey;'>{version}</p>", unsafe_allow_html=True)
    st.subheader("Koristeći LangChain i Streamlit...")
    with st.expander("Pročitajte uputstvo:"):
        st.caption("""
                   Prethodni korak bio je kreiranje pitanja. To smo radili pomocu besplatnog CHATGPT modela. Iz svake oblasti (ili iz dokumenta) zamolimo CHATGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta.\n
                   Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka. Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponasanja modela) i naziv FT modela. Kliknite na Submit i sacekajte da se obrada zavrsi. Fajl sa odgovorima cete kasnije korisiti za kreiranje FT modela.\n
                   Pre prelaska na sledecu fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.
                   """)
        st.divider()

    # Initialize ChatOpenAI and RetrievalQA

    st.session_state["izlaz"] = ""
    model, temp = init_cond_llm()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model, temperature=temp)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), verbose=False
    )

    # Save the user input in the session state
    placeholder = st.empty()
    st.session_state["task"] = ""

    # Create a form with a text input and a submit button
    with placeholder.form(key="my_form", clear_on_submit=True):
        query = (
            "Using langchain and streamlite, "
            + st.text_area(
                label="Detaljno opišite šta želite da uradim (kod, objašnjenje ili sl): ",
                key="1",
                value=st.session_state["task"],
                help="Npr. Napravi kod koji će da ispiše Hello World!",
            )
            + "."
        )
        submit_button = st.form_submit_button(
            label="Submit", help="Kliknite ovde da pokrenete izvršavanje"
        )

        # If the submit button is clicked, clear the session state and run the query
        if submit_button:
            st.session_state["task"] = ""
            with st.spinner("Sačekajte trenutak..."):
                st.session_state["izlaz"] = qa.run(query)
                st.write(st.session_state["izlaz"])

    if "izlaz" in st.session_state:
        st.download_button(
            "Download as .txt",
            st.session_state["izlaz"],
            file_name="koder.txt",
            help="Kliknite ovde da preuzmete fajl",
        )


# koristi se samo za deployment na streamlit cloudu
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
