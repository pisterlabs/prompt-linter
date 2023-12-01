# program za pisanje u stilu neke osobe, uzima stil i temu iz Pinecone indexa

# uvoze se biblioteke
import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from html2docx import html2docx
import markdown
import pdfkit
from myfunc.mojafunkcija import st_style, positive_login, init_cond_llm

# glavna funkcija

version = "14.10.23. Self"


def main():
    if "text" not in st.session_state:
        st.session_state.text = "text"
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = ""
    # Retrieving API keys from env
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"],
    )

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Define metadata fields
    metadata_field_info = [
        AttributeInfo(name="title", description="Tema dokumenta", type="string"),
        AttributeInfo(name="keyword", description="reci za pretragu", type="string"),
        AttributeInfo(
            name="text", description="The Content of the document", type="string"
        ),
        AttributeInfo(
            name="source", description="The Source of the document", type="string"
        ),
    ]

    # Define document content description
    document_content_description = "Sistematizacija radnih mesta"

    # Initialize OpenAI embeddings and LLM and all variables
    model, temp = init_cond_llm()
    llm = ChatOpenAI(model_name=model, temperature=temp, openai_api_key=openai_api_key)

    if "namespace" not in st.session_state:
        st.session_state.namespace = "sistematizacija3"
    if "index_name" not in st.session_state:
        st.session_state.index_name = "embedings1"

    # Izbor stila i teme
    st.subheader("Using Self Query")

    vectorstore = Pinecone.from_existing_index(
        st.session_state.index_name,
        embeddings,
        st.session_state.text,
        namespace=st.session_state.namespace,
    )
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True,
    )

    # Prompt template - Loading text from the file

    with st.form(key="stilovi", clear_on_submit=False):
        # izbor teme
        zahtev = st.text_area(
            "Postavite pitanje: ",
            key="prompt_prva",
            height=150,
        )

        submit_button = st.form_submit_button(label="Submit")

        # pocinje obrada, prvo se pronalazi tematika, zatim stil i na kraju se generise odgovor

        if submit_button:
            with st.spinner("Obradjujem temu..."):
                docs = retriever.get_relevant_documents(zahtev)
                prompt = f"Relevant documents: {docs}\n\nBased on the documents, answer the question: {zahtev}"
                # zameniti predict za llmchain
                with st.expander("PROMPT", expanded=False):
                    st.write(prompt)

                try:
                    st.session_state.odgovor = llm.predict(prompt)

                    # Izrada verzija tekstova za fajlove formnata po izboru
                    with st.expander("FINALNI TEKST", expanded=True):
                        st.markdown(st.session_state.odgovor)
                except Exception as e:
                    st.warning(
                        f"Nisam u mogucnosti za zavrsim tekst. Pokusajte sa modelom koji ima veci kontekst. {e}"
                    )
    # html to docx
    html = markdown.markdown(st.session_state.odgovor)
    buf = html2docx(html, title="Zapisnik")
    # create pdf
    options = {
        "encoding": "UTF-8",  # Set the encoding to UTF-8
        "no-outline": None,
        "quiet": "",
    }
    pdf_data = pdfkit.from_string(html, False, options=options)

    with st.sidebar:
        st.download_button(
            "Download TekstuStilu.txt",
            st.session_state.odgovor,
            file_name="TekstuStilu.txt",
        )
        st.download_button(
            label="Download TekstuStilu.pdf",
            data=pdf_data,
            file_name="TekstuStilu.pdf",
            mime="application/octet-stream",
        )
        st.download_button(
            label="Download TekstuStilu.docx",
            data=buf.getvalue(),
            file_name="TekstuStilu.docx",
            mime="docx",
        )


# Login
st_style()
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
