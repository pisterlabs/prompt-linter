# This code does summarization

# Importing necessary modules
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import HumanMessage, SystemMessage

# from langchain import LLMChain
import streamlit as st
import os
from html2docx import html2docx
import markdown
import openai
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    open_file,
    init_cond_llm,
    greska,
    show_logo,
    def_chunk,
)

# from random import randint
# from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
# from langchain.schema.runnable import RunnableConfig
# from langsmith import Client
# from streamlit_feedback import streamlit_feedback
# from langchain.callbacks.tracers.langchain import wait_for_all_tracers

import pdfkit
import PyPDF2
import re
import io

# these are the environment variables that need to be set for LangSmith to work
# os.environ["LANGCHAIN_PROJECT"] = "Zapisnik"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
# os.environ.get("LANGCHAIN_API_KEY")

version = "28.10.23."

st.set_page_config(page_title="Zapisnik", page_icon="👉", layout="wide")
st_style()


def main():
    # client = Client()
    side_zapisnik()
    # Read OpenAI API key from envtekst za
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # initial prompt
    prompt_string = open_file("prompt_summarizer.txt")
    prompt_string_pam = open_file("prompt_pam.txt")
    opis = "opis"
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Zapisnik ✍️")  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo 🧜"):
        st.caption(
            """
                   Služi za generisanje sažetaka transkripta sastanaka - učitate mu sami transkript
                   (to mora da se učita, ne može da se kuca direktno na sajtu) - onda ili ukucavate ili učitavate promptove.\n
                   Promptovima govorite kako bi hteli da se vrši sumarizacija (koliko detaljno, na šta da se fokusira ili
                   šta da izbegava, itd.) i kako bi hteo da je strukturiran/formatiran izlazni tekst.\n
                   Promptove možete naći na Public-u - folder AI Dev.
                   """
        )
        st.image(
            "https://test.georgemposi.com/wp-content/uploads/2023/09/Zapisnik1.png"
        )
        st.caption(
            """\n
                   1.	Odabir modela i temperature (isto kao i kod Multi Tool Chatbot-a)\n
                   2.	Uploadovanje teksta koji biste da sumarizujete.\n
                   3.	Uploadovanje ili ručno unošenje promptova koje primenjujete nad tim tekstom.
                   Isto kao i kod Pisi u stilu FT, aplikacija gleda ono što je u tekstualnom polju.\n
                   4.	Polje za unos komentara, nakon izvršavanja programa.
                   Napomena: unos komentara i ocenjivanje kod Zapisnika je potpuno analogno onom za Pisi u stilu FT.\n
                   Pojašnjenje:\n
                   Postoje dva prompta: početni i finalni - razlog je to što program deli tekst na više celina
                   koje se potom obrađuju pojedinačno, pa kasnije kombinuju u jednu celinu, koja se ispisuje u aplikaciji.
                   Za sada se pokazalo da dobijamo bolje rezultate ako više forsiramo instrukcije za sumarizaciju kroz početni prompt.
                   """
        )
        st.image(
            "https://test.georgemposi.com/wp-content/uploads/2023/09/Zapisnik2.png"
        )
        st.caption(
            """\n
                   1.	Izlaz iz programa - sumarizovan tekst i opcije za download-ovanje.\n
                   2.	Komentar na sumarizovan tekst (nakon kliktanja na Enter).\n
                   3.	Ocenjivanje (1 - 5).\n
                   4.	Polje za unos komentara je sada zaključano - za novu iteraciju je najbolje uraditi refresh stranice.\n
                   Napomena:\n
                   Ova aplikacija trenutno nije predviđena za iterativnu upotrebu.
                   """
        )
    st.caption(
        """
               U svrhe testiranja možete birati GPT 4 (8K) ili GPT 3.5 Turbo (16k) modele.\n
               Date su standardne instrukcije koji mozete promeniti po potrebi. Promptove možete čuvati i uploado-vati u txt formatu.\n
               * Dokumenti veličine do 5000 karaktera će biti tretirani kao jedan. Dozvoljeni formati su txt, docx i pdf.
               """
    )

    uploaded_file = st.file_uploader(
        "Izaberite tekst za sumarizaciju",
        key="upload_file",
        type=["txt", "pdf", "docx"],
    )

    if "dld" not in st.session_state:
        st.session_state.dld = "Zapisnik"

    # markdown to html
    html = markdown.markdown(st.session_state.dld)
    # html to docx
    buf = html2docx(html, title="Zapisnik")

    options = {
        "encoding": "UTF-8",  # Set the encoding to UTF-8
        "no-outline": None,
        "quiet": "",
    }

    pdf_data = pdfkit.from_string(html, cover_first=False, options=options)

    # summarize chosen file
    if uploaded_file is not None:
        model, temp = init_cond_llm(1)
        # Initializing ChatOpenAI model
        llm = ChatOpenAI(
            model_name=model, temperature=temp, openai_api_key=openai.api_key
        )

        prva_file = st.file_uploader(
            "Izaberite početni prompt koji možete editovati ili pišite prompt od pocetka",
            key="upload_prva",
            type="txt",
        )
        if prva_file is not None:
            prva = prva_file.getvalue().decode("utf-8")  # Loading text from the file
        else:
            prva = " "

        druga_file = st.file_uploader(
            "Izaberite finalni prompt koji možete editovati ili pišite prompt od početka",
            key="upload_druga",
            type="txt",
        )
        if druga_file is not None:
            druga = druga_file.getvalue().decode("utf-8")  # Loading text from the file
        else:
            druga = " "

        with io.open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getbuffer())

        if ".pdf" in uploaded_file.name:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)
            text_content = ""

            for page in range(num_pages):
                page_obj = pdf_reader.pages[page]
                text_content += page_obj.extract_text()
            text_content = text_content.replace("•", "")
            text_content = re.sub(r"(?<=\b\w) (?=\w\b)", "", text_content)
            with io.open("temp.txt", "w", encoding="utf-8") as f:
                f.write(text_content)

            loader = UnstructuredFileLoader("temp.txt", encoding="utf-8")
        else:
            # Creating a file loader object
            loader = UnstructuredFileLoader(uploaded_file.name, encoding="utf-8")

        result = loader.load()
        chunk_size = 5000
        chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )  # Creating a text splitter object
        duzinafajla = len(result[0].page_content)

        # Splitting the loaded text into smaller chunks
        texts = text_splitter.split_documents(result)
        chunkova = len(texts)
        st.success(
            f"Tekst je dugačak {duzinafajla} karaktera i podeljen je u {chunkova} delova."
        )
        if chunkova == 1:
            st.info(
                "Tekst je kratak i biće obrađen u celini koristeći samo drugi prompt."
            )

        out_elements = [
            "Zapisnik -",
            "m=" + model.rsplit("-", 1)[-1],
            "t=" + str(temp),
            "chunk s_o=" + str(chunk_size / 1000) + "k_" + str(chunk_overlap),
        ]
        out_name = " ".join(out_elements)

        with st.form(key="my_form", clear_on_submit=False):
            opis = st.text_area(
                "Unesite instrukcije za početnu sumarizaciju (kreiranje više manjih delova teksta): ",
                prva,
                key="prompt_prva",
                height=150,
            )

            opis_kraj = st.text_area(
                "Unesite instrukcije za finalnu sumarizaciju (kreiranje finalne verzije teksta): ",
                druga,
                key="prompt_druga",
                height=150,
            )

            PROMPT = PromptTemplate(
                template=prompt_string, input_variables=["text", "opis"]
            )  # Creating a prompt template object
            PROMPT_pam = PromptTemplate(
                template=prompt_string_pam, input_variables=["text", "opis_kraj"]
            )  # Creating a prompt template object
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                with st.spinner("Sačekajte trenutak..."):
                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        verbose=True,
                        map_prompt=PROMPT,
                        combine_prompt=PROMPT_pam,
                        token_max=4000,
                    )
                    # Load the summarization chain with verbose mode

                    suma = AIMessage(
                        content=chain.run(
                            input_documents=texts, opis=opis, opis_kraj=opis_kraj
                        )
                    )

                    st.session_state.dld = suma.content
                    html = markdown.markdown(st.session_state.dld)
                    buf = html2docx(html, title="Zapisnik")

                    pdf_data = pdfkit.from_string(html, False, options=options)

        if st.session_state.dld != "Zapisnik":
            st.write("Download-ujte vaše promptove")
            col4, col5 = st.columns(2)
            with col4:
                st.download_button(
                    "Download prompt 1 as .txt", opis, file_name="prompt1.txt"
                )
            with col5:
                st.download_button(
                    "Download prompt 2 as .txt", opis_kraj, file_name="prompt2.txt"
                )
            st.write("Download-ujte vas zapisnik")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download Zapisnik as .txt",
                    st.session_state.dld,
                    file_name=out_name + ".txt",
                )
            with col2:
                st.download_button(
                    label="Download Zapisnik as .docx",
                    data=buf.getvalue(),
                    file_name=out_name + ".docx",
                    mime="docx",
                )
            with col3:
                st.download_button(
                    label="Download Zapisnik as .pdf",
                    data=pdf_data,
                    file_name=out_name + ".pdf",
                    mime="application/octet-stream",
                )
            with st.expander("Sažetak", True):
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary

    # if prompt := st.chat_input(placeholder="Unesite komentare na rad programa."):
    #     st.session_state["user_feedback"] = prompt
    #     st.chat_input(placeholder="Feedback je sačuvan!", disabled=True)
    #     st.session_state.feedback = None
    #     st.session_state.feedback_update = None
    #     run_collector = RunCollectorCallbackHandler()

    #     prompt = ChatPromptTemplate.from_messages([("system", "Hi"), ("human", "Hi")])
    #     llm = ChatOpenAI(temperature=0.7)
    #     chain = LLMChain(prompt=prompt, llm=llm)

    #     x = chain.invoke(
    #         {"input": "Hi."},
    #         config=RunnableConfig(
    #             callbacks=[run_collector],
    #             tags=["Streamlit Chat"],
    #         ),
    #     )["text"]

    #     run = run_collector.traced_runs[0]
    #     run_collector.traced_runs = []
    #     st.session_state.run_id = run.id
    #     wait_for_all_tracers()
    #     try:
    #         client.share_run(run.id)
    #     except ValueError:
    #         st.write("...")

    # if st.session_state.get("run_id"):
    #     with st.chat_message("assistant", avatar="🤖"):
    #         message_placeholder = st.empty()
    #         message_placeholder.markdown(
    #             ":rainbow[Samo još ocenite od 1 do 5 dobijene rezultate.]"
    #         )
    #     feedback = streamlit_feedback(
    #         feedback_type="faces", key=f"feedback_{st.session_state.run_id}"
    #     )
    #     scores = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}
    #     if feedback:
    #         score = scores[feedback["score"]]
    #         feedback = client.create_feedback(
    #             st.session_state.run_id,
    #             "ocena",
    #             score=score,
    #             comment=st.session_state["user_feedback"],
    #         )
    #         st.session_state.feedback = {
    #             "feedback_id": str(feedback.id),
    #             "score": score,
    #         }

    # if st.session_state.get("feedback"):
    #     feedback = st.session_state.get("feedback")
    #     x = ["🎭", "🐯", "👺", "👻", "😸", "🤓", "🤡", "🦄", "🧟‍♀️", "☘️"]
    #     st.write(
    #         f"{x[randint(0, len(x) - 1)]} Ova aplikacija NE radi iterativno - mora refresh stranice!"
    #     )
    #     st.chat_input(placeholder="To je to - hvala puno!", disabled=True)


def korekcija_transkripta():
    uploaded_file = st.file_uploader("Transkript", type=["txt"])
    ceo_text = " "
    if uploaded_file is not None:
        with st.form(key="ispravi_transkript"):
            submit_button = st.form_submit_button(
                label="Ispravi transkript",
                help="Ispravlja transkript na srpskom jeziku",
            )
            if submit_button:
                transkript = uploaded_file.getvalue().decode("utf-8")
                text_splitter = RecursiveCharacterTextSplitter(
                    # Set a really small chunk size, just to show.
                    chunk_size=10000,
                    chunk_overlap=0,
                    length_function=len,
                    is_separator_regex=False,
                )

                texts = text_splitter.create_documents([transkript])
                chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
                ceo_text = " "

                with st.spinner("Ispravlja se transkript..."):
                    placeholder = st.empty()
                    total_texts = len(texts)
                    for i, text in enumerate(texts, start=1):
                        with placeholder:
                            st.info(f"Obradjuje se {i} od {total_texts} strana")
                        txt = text.page_content
                        messages = [
                            SystemMessage(
                                content="You are the Serbian language expert. you must fix grammar and spelling errors but otherwise keep the text as is, in the Serbian language."
                            ),
                            HumanMessage(content=txt),
                        ]
                        odgovor = chat.invoke(messages).content
                        ceo_text = ceo_text + " " + odgovor

                    # na kraju ih sve slepi u jedan dokument i sacuva ga u fajl
                    with placeholder:
                        st.success("Zavrsena obrada transkripta")
        if ceo_text != " ":
            st.download_button(
                "Preuzmite ispravljen transkript",
                ceo_text,
                file_name=f"korigovan_{uploaded_file.name}",
            )


def korekcija_imena():
    if "result_string" not in st.session_state:
        st.session_state.result_string = ""
    file_name = " "
    with st.sidebar:
        st.info(
            "Korekcija imena učesnika sastanka. Radi sa gpt-4, temp=0, overlap treba da je 0. Ne cuva formatiranje teksta."
        )
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        model = "gpt-4"
        temp = 0
        chat = ChatOpenAI(model=model, temperature=temp, openai_api_key=openai_api_key)
        template = (
            "You are a helpful assistant that fixes misspelled names in transcript."
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        result_string = ""
        prompt = PromptTemplate(
            template="""Please only fix the names of the people mentioned that are misspelled in this text: 
            {text} 
            
            The correct names are {ucesnici}. 
            
            Do not write any comment, just the original text with corrected names. If there are no corrections to be made, just write the original text again.""",
            input_variables=["ucesnici", "text"],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        dokum = st.file_uploader(
            "Izaberite .txt",
            key="upload_file_fix_names",
            type=["txt", "pdf", "docx"],
            help="Izaberite fajl za korekciju imena",
        )

        if dokum:
            with io.open(dokum.name, "wb") as file:
                file.write(dokum.getbuffer())
            loader = UnstructuredFileLoader(dokum.name, encoding="utf-8")

            data = loader.load()
            chunk_size, chunk_overlap = def_chunk()
            # Split the document into smaller parts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            texts = text_splitter.split_documents(data)
            new_text = []
            for txt in texts:
                new_text.append(txt.page_content)
            with st.form(key="imena"):
                ucesnici = st.text_input(
                    "Unesite imena učesnika: ",
                    help="Imena učesnika sastanka odvojiti zarezom i razmakom",
                )
                submit = st.form_submit_button(
                    label="Submit", help="Pokreće izvršenje programa"
                )
                if submit:
                    with st.spinner("Obrada teksta u toku..."):
                        # Get a chat completion from the formatted messages
                        for text in new_text:
                            result = chat(
                                chat_prompt.format_prompt(
                                    ucesnici=ucesnici,
                                    text=text,
                                ).to_messages()
                            )
                            result_string += result.content
                            result_string = result_string.replace("\n", " ")

                        with st.expander("Obrađen tekst"):
                            st.write(result_string)
                        st.session_state.result_string = result_string
            if st.session_state.result_string != "":
                html = markdown.markdown(st.session_state.result_string)
                buf = html2docx(html, title="Zapisnik")

                options = {
                    "encoding": "UTF-8",  # Set the encoding to UTF-8
                    "no-outline": None,
                    "quiet": "",
                }

                pdf_data = pdfkit.from_string(html, cover_first=False, options=options)
                name_without_extension = os.path.splitext(dokum.name)[0]
                skinuto = st.download_button(
                    "Download .txt",
                    data=st.session_state.result_string,
                    file_name=f"fix_{name_without_extension}.txt",
                    help="Download obradjenog dokumenta",
                )
                skinuto = st.download_button(
                    label="Download .docx",
                    data=buf.getvalue(),
                    file_name=f"fix_{name_without_extension}.docx",
                    mime="docx",
                )
                skinuto = st.download_button(
                    label="Download .pdf",
                    data=pdf_data,
                    file_name=f"fix_{name_without_extension}.pdf",
                    mime="application/octet-stream",
                )
                if skinuto:
                    st.success(f"Tekstovi sačuvani na {file_name}")


def transkript():
    # Read OpenAI API key from env
    from openai import OpenAI
    client = OpenAI()
    with st.sidebar:  # App start
        st.info("Konvertujte MP3 u TXT")
        audio_file = st.file_uploader(
            "Max 25Mb",
            type="mp3",
            key="audio_",
            help="Odabir dokumenta",
        )
        # transcript_json= "transcript"
        transcritpt_text = "transcript"
        if audio_file is not None:
            placeholder = st.empty()
            st.session_state["question"] = ""

            with placeholder.form(key="my_jezik", clear_on_submit=False):
                jezik = st.selectbox(
                    "Odaberite jezik izvornog teksta 👉",
                    (
                        "sr",
                        "en",
                        "th",
                        "de",
                        "fr",
                        "hu",
                        "it",
                        "ja",
                        "ko",
                        "pt",
                        "ru",
                        "es",
                        "zh",
                    ),
                    key="jezik",
                    help="Odabir jezika",
                )

                submit_button = st.form_submit_button(label="Submit")

                if submit_button:
                    with st.spinner("Sačekajte trenutak..."):
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file, language=jezik
                        )
                        # transcript_dict = {"text": transcript.text}
                        transcritpt_text = transcript.text
                        with st.expander("Transkript"):
                            # Create an expander in the Streamlit application with label 'Koraci'
                            st.info(transcritpt_text)
                            # Display the intermediate steps inside the expander
            if transcritpt_text is not None:
                st.download_button(
                    "Download transcript",
                    transcritpt_text,
                    file_name="transcript.txt",
                    help="Odabir dokumenta",
                )


def side_zapisnik():
    with st.sidebar:
        izbor_app = st.selectbox(
            "Izaberite pomoćnu akciju",
            ("Transkript", "Korekcija transkripta", "Korekcija imena"),
            help="Odabir akcije za pripremu zapisnika",
        )
        if izbor_app == "Transkript":
            transkript()
        elif izbor_app == "Korekcija transkripta":
            korekcija_transkripta()
        elif izbor_app == "Korekcija imena":
            korekcija_imena()


# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
