# This code does summarization

# Importing necessary modules
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate
import streamlit as st
import os
from html2docx import html2docx
import markdown
import openai
from mojafunkcja import st_style, positive_login, open_file, init_cond_llm, greska

from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from vanilla_chain import get_llm_chain
client = Client()

# from pdfquery import PDFQuery
from xhtml2pdf import pisa
import PyPDF2
import re
import io

# these are the environment variables that need to be set for LangSmith to work
os.environ["LANGCHAIN_PROJECT"] = "Zapisnik"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ.get("LANGCHAIN_API_KEY")

st.set_page_config(
    page_title="Zapisnik",
    page_icon="👉",
    layout="wide"
)
st_style()

def main():
    # Read OpenAI API key from envtekst za
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    # initial prompt
    prompt_string = open_file("prompt_summarizer.txt")
    prompt_string_pam = open_file("prompt_pam.txt")
    opis = "opis"
    st.subheader('Zapisnik ✍️')  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo 🧜"):
        st.caption("""
                   Služi za generisanje sažetaka transkripta sastanaka - učitate mu sami transkript
                   (to mora da se učita, ne može da se kuca direktno na sajtu) - onda ili ukucavate ili učitavate promptove.\n
                   Promptovima govorite kako bi hteli da se vrši sumarizacija (koliko detaljno, na šta da se fokusira ili
                   šta da izbegava, itd.) i kako bi hteo da je strukturiran/formatiran izlazni tekst.\n
                   Promptove možete naći na Public-u - folder AI Dev.
                   """)
        st.image("https://test.georgemposi.com/wp-content/uploads/2023/09/Zapisnik1.png")
        st.caption("""\n
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
                   """)
        st.image("https://test.georgemposi.com/wp-content/uploads/2023/09/Zapisnik2.png")
        st.caption("""\n
                   1.	Izlaz iz programa - sumarizovan tekst i opcije za download-ovanje.\n
                   2.	Komentar na sumarizovan tekst (nakon kliktanja na Enter).\n
                   3.	Ocenjivanje (1 - 5).\n
                   4.	Polje za unos komentara je sada zaključano - za novu iteraciju je najbolje uraditi refresh stranice.\n
                   Napomena:\n
                   Ova aplikacija trenutno nije predviđena za iterativnu upotrebu.
                   """)
    st.caption("""
               U svrhe testiranja možete birati GPT 4 (8K) ili GPT 3.5 Turbo (16k) modele.\n
               Date su standardne instrukcije koji mozete promeniti po potrebi. Promptove možete čuvati i uploado-vati u txt formatu.\n
               * Dokumenti veličine do 5000 karaktera će biti tretirani kao jedan. Dozvoljeni formati su txt, docx i pdf.
               """)

    uploaded_file = st.file_uploader(
        "Izaberite tekst za sumarizaciju", key="upload_file", type=['txt', 'pdf', 'docx'])

    if 'dld' not in st.session_state:
        st.session_state.dld = "Zapisnik"
    
    # markdown to html
    html = markdown.markdown(st.session_state.dld)
    # html to docx
    buf = html2docx(html, title="Zapisnik")

    # pdf_data = pdfkit.from_string(html, cover_first=False, options=options)

    html1 = """
    <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @font-face {
                    font-family: Arial;
                    src: url(path/to/arial.ttf); /* Replace with the actual path to your font file */
                }
                body {
                    font-family: Arial, sans-serif;
                }
            </style>
        </head>
        <body>
            <p>
            """ + markdown.markdown(st.session_state.dld) + """
            </p>
        </body>
    </html>
    """

    x = """
    html_io = io.BytesIO(html.encode('UTF-8'))
    pdf_data = io.BytesIO()
    pisa.CreatePDF(html_io, pdf_data, encoding='UTF-8', embed_font=True)
    pdf_data.seek(0)
    pdf_data = pdf_data.getvalue()
    st.write(pdf_data)
    """

    # summarize chosen file
    if uploaded_file is not None:
        model, temp = init_cond_llm()
        # Initializing ChatOpenAI model
        llm = ChatOpenAI(model_name=model, temperature=temp,
                         openai_api_key=openai.api_key)

        prva_file = st.file_uploader(
            "Izaberite početni prompt koji možete editovati ili pišite prompt od pocetka", key="upload_prva", type='txt')
        if prva_file is not None:
            prva = prva_file.getvalue().decode("utf-8")  # Loading text from the file
        else:
            prva = " "
        
        druga_file = st.file_uploader(
            "Izaberite finalni prompt koji možete editovati ili pišite prompt od početka", key="upload_druga", type='txt')
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
            text_content = text_content.replace('•', '')
            text_content = re.sub(r'(?<=\b\w) (?=\w\b)', '', text_content)
            with io.open('temp.txt', 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            loader = UnstructuredFileLoader('temp.txt', encoding="utf-8")
        else:
            # Creating a file loader object
            loader = UnstructuredFileLoader(
                uploaded_file.name, encoding="utf-8")


        result = loader.load() 
        chunk_size = 5000
        chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Creating a text splitter object
        duzinafajla = len(result[0].page_content)

        # Splitting the loaded text into smaller chunks
        texts = text_splitter.split_documents(result)
        chunkova = len(texts)
        st.success(
            f"Tekst je dugačak {duzinafajla} karaktera i podeljen je u {chunkova} delova.")
        if chunkova == 1:
            st.info(
                "Tekst je kratak i biće obradjen u celini koristeći samo drugi prompt")

        out_elements = ["Zapisnik -", 
                        "m=" + model.rsplit('-', 1)[-1], 
                        "t=" + str(temp), 
                        "chunk s_o=" + str(chunk_size/1000) + "k_" + str(chunk_overlap)]
        out_name = ' '.join(out_elements)

        with st.form(key='my_form', clear_on_submit=False):

            opis = st.text_area("Unesite instrukcije za početnu sumarizaciju (kreiranje više manjih delova teksta): ",
                                prva,
                                key="prompt_prva", height=150)

            opis_kraj = st.text_area("Unesite instrukcije za finalnu sumarizaciju (kreiranje finalne verzije teksta): ",
                                     druga,
                                     key="prompt_druga", height=150)

            PROMPT = PromptTemplate(template=prompt_string, input_variables=[
                                    "text", "opis"])  # Creating a prompt template object
            PROMPT_pam = PromptTemplate(template=prompt_string_pam, input_variables=[
                                        "text", "opis_kraj"])  # Creating a prompt template object
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                with st.spinner("Sačekajte trenutak..."):
                    chain = load_summarize_chain(
                        llm, chain_type="map_reduce", verbose=True, map_prompt=PROMPT, combine_prompt=PROMPT_pam, token_max=4000)
                    # Load the summarization chain with verbose mode
                    try:
                        suma = AIMessage(content=chain.run(
                            input_documents=texts, opis=opis, opis_kraj=opis_kraj))

                        st.session_state.dld = suma.content
                        html = markdown.markdown(st.session_state.dld)
                        buf = html2docx(html, title="Zapisnik")
                        pdf_data = pisa.CreatePDF(src=html, encoding='utf-8')
                    except Exception as e:
                        greska(e)

        if st.session_state.dld != "Zapisnik":
            st.write("Download-ujte vaše promptove")
            col4, col5 = st.columns(2)
            with col4:
                st.download_button("Download prompt 1 as .txt",
                                   opis, file_name="prompt1.txt")
            with col5:

                st.download_button("Download prompt 2 as .txt",
                                   opis_kraj, file_name="prompt2.txt")
            st.write("Download-ujte vas zapisnik")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Zapisnik as .txt",
                                   st.session_state.dld, file_name=out_name + ".txt")
            with col2:
                st.download_button(label="Download Zapisnik as .docx",
                                   data=buf.getvalue(),
                                   file_name=out_name + ".docx",
                                   mime="docx")
            x = """ izmeni i poziv st.columns iznad
            with col3:
                st.download_button(label="Download Zapisnik as .pdf",
                                   data=pdf_data,
                                   file_name=out_name + ".pdf",
                                   mime='application/octet-stream')
            """
            with st.expander('Sažetak', True):
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary


        if prompt := st.chat_input(placeholder="Unesite sve napomene/komentare koje imate u vezi sa performansama programa."):
            st.chat_message("user", avatar="👽").write(prompt)
            st.session_state['user_feedback'] = prompt
            st.chat_input(placeholder="Vaš feedback je sačuvan!", disabled=True)
            st.session_state.feedback = None
            st.session_state.feedback_update = None
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Samo sekund!")
                run_collector = RunCollectorCallbackHandler()
                message_placeholder.markdown("Samo još ocenite od 1 do 5 dobijene rezultate.")
                    
                memory = ConversationBufferMemory(
                    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
                    return_messages=True,
                    memory_key="chat_history",
                )
                
                chain = get_llm_chain("Hi", memory)

                x = chain.invoke(
                    {"input": "Hi."}, config=RunnableConfig(
                    callbacks=[run_collector], tags=["Streamlit Chat"],)
                    )["text"]
                
                message_placeholder.markdown("Samo još ocenite od 1 do 5 dobijene rezultate.")
                run = run_collector.traced_runs[0]
                run_collector.traced_runs = []
                st.session_state.run_id = run.id
                wait_for_all_tracers()
                client.share_run(run.id)

        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(feedback_type="faces", key=f"feedback_{st.session_state.run_id}",)
            scores = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}
            if feedback:
                score = scores[feedback["score"]]
                feedback = client.create_feedback(st.session_state.run_id, "ocena", score=score)
                st.session_state.feedback = {"feedback_id": str(feedback.id), "score": score}

        if st.session_state.get("feedback"):
            feedback = st.session_state.get("feedback")
            feedback_id = feedback["feedback_id"]
            score = feedback["score"]

            st.session_state.feedback_update = {
                "comment": st.session_state['user_feedback'],
                "feedback_id": feedback_id,
            }
            client.update_feedback(feedback_id)
            st.chat_input(placeholder="To je to - hvala puno!", disabled=True)

        if st.session_state.get("feedback_update"):
            feedback_update = st.session_state.get("feedback_update")
            feedback_id = feedback_update.pop("feedback_id")
            client.update_feedback(feedback_id, **feedback_update)
            st.session_state.feedback = None
            st.session_state.feedback_update = None

name, authentication_status, username = positive_login(main, "08.09.23. - Nemanja")
