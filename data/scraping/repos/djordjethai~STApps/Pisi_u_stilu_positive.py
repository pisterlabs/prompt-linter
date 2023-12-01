# program za pisanje u stilu neke osobe, uzima stil i temu iz Pinecone indexa

# uvoze se biblioteke
import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from html2docx import html2docx
import markdown
import pdfkit
from mojafunkcja import st_style, positive_login, open_file, init_cond_llm
# from langchain.output_parsers import CommaSeparatedListOutputParser


# Zaglavlje stranice
st.set_page_config(
    page_title="Pisi u stilu",
    page_icon="👉",
    layout="wide"
)

# glavna funkcija


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
        AttributeInfo(name="chunk",
                      description="The number of the document chunk", type="int"),
        AttributeInfo(
            name="url", description="The name of url or document", type="string"),
        AttributeInfo(
            name="source", description="The source of the document", type="string"),
    ]

    # Define document content description
    document_content_description = "Content of the document"

    # Initialize OpenAI embeddings and LLM and all variables
    model, temp = init_cond_llm()
    llm = ChatOpenAI(model_name=model, temperature=temp,
                     openai_api_key=openai_api_key)
    # output_parser = CommaSeparatedListOutputParser()

    if "text" not in st.session_state:
        st.session_state.text = "text"
    if "namespace" not in st.session_state:
        st.session_state.namespace = "positive"
    if "index_name" not in st.session_state:
        st.session_state.index_name = "embedings1"
    if "stil" not in st.session_state:
        st.session_state.stil = " "
    if "text_stil" not in st.session_state:
        st.session_state.text_stil = " "
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = " "
    if "priprema" not in st.session_state:
        st.session_state.priprema = " "
    if "pred_odgovor" not in st.session_state:
        st.session_state.pred_odgovor = " "
    if "stil_odgovor" not in st.session_state:
        st.session_state.stil_odgovor = " "
    if "prompt" not in st.session_state:
        st.session_state.prompt = " "
    if "description_prompt" not in st.session_state:
        st.session_state.description_prompt = " "
    if "tematika" not in st.session_state:
        st.session_state.tematika = " "

    # Izbor stila i teme
    st.subheader('Pišite u stilu indeksiranih osoba 🌆')
    st.caption("""
               Ova aplikacija omogućava da se pronađe tekst određene osobe na određenu temu, 
               i da se koristi kao osnova za pisanje teksta u stilu te osobe.\n
               Kada bude bilo dovoljno osoba sa svojim stilovima, stil se moze odrediti na osnovu imena prijavljenog korisnika.
               """)

# The SelfQueryRetriever will use the LLM to expand the original query into a richer, more semantic query
# before searching the vectorstore including meta data information.
#
# MMR provides an effective and simple way to improve search result diversity and reduce redundancy
# compared to standard ranking techniques. It balances relevance and diversity without extensive
# re-engineering of the retrieval system.

    vectorstore = Pinecone.from_existing_index(
        st.session_state.index_name, embeddings, st.session_state.text, namespace=st.session_state.namespace)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, enable_limit=True, verbose=True)
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr", verbose=True)
    # Prompt template - Loading text from the file
    prompt_file = st.file_uploader(
        "Za opis teme, izaberite početni prompt koji možete editovati ili pišite prompt od početka", key="upload_prompt", type='txt')
    if prompt_file is not None:
        st.session_state.prompt = prompt_file.getvalue().decode("utf-8")
    description_file = st.file_uploader(
        "Za opis stila, izaberite početni prompt koji možete editovati ili pišite prompt od početka", key="upload_prompt_opis", type='txt')
    if description_file is not None:
        st.session_state.description_prompt = open_file(
            description_file.name)
    with st.form(key='stilovi', clear_on_submit=False):
        # izbor teme
        zahtev = st.text_area("Opišite temu, iz oblasti Positive, ili opšte teme: ",
                              st.session_state.prompt,
                              key="prompt_prva", height=150)
    # izbor osobe
        izbor_osobe = st.selectbox(
            "Odaberite osobu:", ("Miljan Radanovic", "Sean Carroll", "Dragan Varagic", "Neuka Osoba", "JJ Zmaj", "Dragan Simic", "Djordje Medakovic"),)
        # ime_osobe = open_file('ime_osobe.txt')
        izbor_stila = open_file('ime_osobe.txt')
        formatted_stil = izbor_stila.format(
            izbor_osobe=izbor_osobe)

        prompt_string = PromptTemplate.from_template(formatted_stil)
    # instrukcije za stil
        oblik = st.text_area("Unesite instrukcije za oblik teksta, dužinu, formatiranje, jezik i sl: ",
                             st.session_state.description_prompt,
                             key="prompt_oblik", height=150)

        submit_button = st.form_submit_button(label='Submit')


# pocinje obrada, prvo se pronalazi tematika, zatim stil i na kraju se generise odgovor

    if submit_button:

        with st.spinner("Obrađujem temu..."):

            # tema init vector_tema, retirever, pred_prompt, pred_odgovor
            #
            #   Na osnovu zahteva za temom,
            #    1. pronaci relevante delove teksta iz indexa i sumirati ih. To ce biti prvi deo krajnjeg prompta
            #    2. na osnovu izbora stila pronaci relevatne tekstove iz indexa i sumirati ih. To ce biti drugi deo krajnjeg prompta
            #    3. spojiti dva odgovora i dodati relevantan tekst za treci upit llm-u, sto ce bit ikrajnji tekst
            #

            # prvo treba pronaci relevantne teme i napisati sumarizaciju
            # Get relevant documents using the retriever
            tematika = st.session_state.tematika = mmr_retriever.get_relevant_documents(
                zahtev)

            # Extract page_content values from the relevant documents
            ukupne_teme = [tema.page_content for tema in tematika]

            # Combine all page_content values with newline separator
            sve_zajedno = "\n".join(ukupne_teme)
            zajedno_prompt = open_file('prompt_tema.txt')
            formatted_string = zajedno_prompt.format(
                zahtev=zahtev, sve_zajedno=sve_zajedno)

            prompt_string = PromptTemplate.from_template(formatted_string)
            # st.write(zajedno_prompt)
            # st.write(formatted_string)
            # Display the combined content

            with st.expander("Tema", expanded=False):
                st.write("Relevantni dokumenti:")
                if sve_zajedno:
                    st.write(formatted_string)
                else:
                    st.write("Nisu pronađeni relevantni dokumenti.")
# ovde treba uraditi rewrite sa nekim dobrim promptom

                llm_chain = LLMChain(prompt=prompt_string, llm=llm)
                odgovor_tema = llm_chain.predict()

        with st.spinner("Tražim stil..."):
            stilovi = st.session_state.stil = retriever.get_relevant_documents(
                formatted_stil)

            # Extract page_content values from the relevant documents
            ukupni_stilovi = [stil.page_content for stil in stilovi]

            # Combine all page_content values with newline separator
            svi_stilovi = "\n".join(ukupni_stilovi)

# ovde mozda ograniciti na k=2
# pa spojiti rewriteovani text teme i osobe sa pazljivo odabranim promptom u stilu
# rewrite this text using only the style from this example ili sl.

            with st.expander("Stil", expanded=False):
                st.write("Relevant styles:")
                if sve_zajedno:
                    st.write(svi_stilovi)
                else:
                    st.write("No relevant documents found.")
                    # prompt_kraj = open_file('prompt_kraj.txt')

                final_prompt = open_file('prompt_kraj.txt')

                formatted_final = final_prompt.format(
                    odgovor_tema=odgovor_tema, zahtev=zahtev, oblik=oblik, svi_stilovi=svi_stilovi)

                prompt_final = PromptTemplate.from_template(formatted_final)

            # Konacan odgovor na osnovu sintetizovanog prompta

                llm_chain = LLMChain(prompt=prompt_final, llm=llm)

            with st.expander("Ceo Prompt", expanded=False):
                st.write(formatted_final)

        with st.spinner("Pišem tekst..."):

            # zameniti predict za llmchain
            try:
                st.session_state.odgovor = llm_chain.predict()

          # Izrada verzija tekstova za fajlove formnata po izboru
                with st.expander("FINALNI TEKST", expanded=True):
                    st.markdown(st.session_state.odgovor)
            except Exception as e:
                st.warning(
                    f"Nisam u mogućnosti za završim tekst. Pokušajte sa modelom koji ima veći kontekst. {e}")
    # html to docx
    html = markdown.markdown(st.session_state.odgovor)
    buf = html2docx(html, title="Zapisnik")
    # create pdf
    options = {
        'encoding': 'UTF-8',  # Set the encoding to UTF-8
        'no-outline': None,
        'quiet': ''
    }
    pdf_data = pdfkit.from_string(html, False, options=options)

    # download
    st.session_state.priprema = "TEMA: " + "\n" + st.session_state.pred_odgovor+"\n" + "STIL: " + "\n" + \
        str(st.session_state.stil)+"\n" + \
        "CEO PROMPT: " "\n" + st.session_state.prompt
    with st.sidebar:
        st.download_button("Download UlazniTekstovi.txt",
                           st.session_state.priprema, file_name="UlazniTekstovi.txt")
        st.download_button("Download TekstuStilu.txt",
                           st.session_state.odgovor, file_name="TekstuStilu.txt")
        st.download_button(label="Download TekstuStilu.pdf",
                           data=pdf_data,
                           file_name="TekstuStilu.pdf",
                           mime='application/octet-stream')
        st.download_button(
            label="Download TekstuStilu.docx",
            data=buf.getvalue(),
            file_name="TekstuStilu.docx",
            mime="docx"
        )


# Login
st_style()
name, authentication_status, username = positive_login(main, "14.08.23")


