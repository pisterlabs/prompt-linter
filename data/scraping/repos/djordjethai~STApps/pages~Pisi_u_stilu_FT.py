# program za pisanje u stilu neke osobe, uzima stil i temu iz Pinecone indexa

# uvoze se biblioteke
import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import LLMChain
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from html2docx import html2docx
import markdown
from mojafunkcja import st_style, positive_login, open_file
from langchain.utilities.google_search import GoogleSearchAPIWrapper

from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from vanilla_chain import get_llm_chain
client = Client()

# from xhtml2pdf import pisa
# import io

# these are the environment variables that need to be set for LangSmith to work
os.environ["LANGCHAIN_PROJECT"] = "Stil"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ.get("LANGCHAIN_API_KEY")


st.set_page_config(
    page_title="Pisi u stilu",
    page_icon="👉",
    layout="wide"
)


def main():
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
    # Retrieving API keys from env
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    # Initialize Pinecone
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
                  environment=os.environ["PINECONE_API_ENV"])
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    search = GoogleSearchAPIWrapper()
    # Initialize OpenAI embeddings and LLM and all variables

    if "model" not in st.session_state:
        st.session_state.model = ""
    if "temp" not in st.session_state:
        st.session_state.temp = 1.0
    if "text" not in st.session_state:
        st.session_state.text = "text"
    if "namespace" not in st.session_state:
        st.session_state.namespace = "koder"
    if "index_name" not in st.session_state:
        st.session_state.index_name = "embedings1"
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = ""
    if "tematika" not in st.session_state:
        st.session_state.tematika = ""
    if "thold" not in st.session_state:
        st.session_state.thold = 0.5
    if "stil" not in st.session_state:
        st.session_state.stil = ""

    # Izbor stila i teme
    st.subheader("Pišite u stilu osoba koje imaju sopstvene Fine-Tunned modele 🏙️")
    with st.expander("Pročitajte uputstvo 🧝"):
        st.caption("""
                   FT se odnosi na Fine-Tuning, tj. prilagođavanje aplikacije nekoj specifičnoj primeni (iliti specijalizacija)
                   - u našem slučaju se aplikacija prilagođava nečijem stilu pisanja (npr. od Miljana).\n
                   Promptove možete naći na Public-u - folder AI Dev.
                   """)
        st.image("https://test.georgemposi.com/wp-content/uploads/2023/09/PisiUStilu1.png")
        st.caption("""\n
                   1.	Parametri za podešavanje rada aplikacije - opisani su u levom meniju, a i intuitivni su.\n
                   2.	Uploadovanje ili direktno kucanje teksta/teme o kojoj biste da pišete.\n
                   Ono što uploadujete će se prikazati u tekstualnom polju ispod - to polje je ono što aplikacija gleda kada se izvršava.\n
                   3.	Ovde je obrnuto u odnosu na Multi Tool Chatbot - prvo se unosi komentar, pa se onda ocenjuje (slika ispod).
                   """)
        st.image("https://test.georgemposi.com/wp-content/uploads/2023/09/PisiUStilu2.png")
        st.caption("""\n
                   1.	Generisani tekst i opcije za skidanje teksta na računar u različitim oblicima.\n
                   2.	Komentar koji ste upisali, pa kliknuli Enter ili strelicu u uglu polja za komentarsanje.\n
                   3.	Ocenjivanje od 1 do 5.\n
                   4.	Polje za unos komentara je sada zaključano - mora refresh stranice da bi se aplikacija opet koristila.\n
                   """)
        
    st.caption("""
               Ova aplikacija omogućava generisanje teksta na određenu temu i da se koristi kao osnova za pisanje teksta u stilu
               odabrane osobe.\n Koristi se Pinecone indeks za pronalaženje teksta na određenu temu.
               Ukoliko ne pronađe odgovarajući tekst, potražiće odgovor na internetu.
               """)
    with st.sidebar:
        st.session_state.namespace = st.selectbox(
            "Odaberite oblast",
            ("koder", "positive"))
        
        ft_model = st.selectbox(
            "Odaberite model",
            ("Dragan Simic", "Miljan Radanovic", "Pera Lozac"))

        if ft_model == "Dragan Simic":
            st.session_state.model = "ft:gpt-3.5-turbo-0613:positive-doo:dragan-simic:7rLzG9Cp"
            st.session_state.stil = "Dragan Simic is an IT expert. He writes in a long sentences in overly polite manner. He always writes in the Serbian language"
        elif ft_model == "Miljan Radanovic":
            st.session_state.model = "ft:gpt-3.5-turbo-0613:positive-doo:miljan:7rIDKWid"
            st.session_state.stil = "Miljan Radanovic is an IT expert. He writes in a long sentences and offten mixes complex and everyday terms in the same sentence. He always writes in the Serbian language"
        elif ft_model == "Pera Lozac":
            st.session_state.model = "ft:gpt-3.5-turbo-0613:positive-doo:pera-lozac:7rKBrShJ"
            st.session_state.stil = "Pera Lozac knows the answers, but he writes in a short sentences in a style of disfluent person and use verbal crutches"

        st.text("", help="Temperatura za stil treba de je što bliže 1.0")
        st.session_state.temp = st.slider(
            'Set temperature (0=strict, 1=creative)', 0.0, 2.0, step=0.1, value=1.0)
        st.text("", help="Relevantnost za temu određuje koji dokmenti će se korsititi iz indeksa. Ako je vrednost 0.0 onda se koriste svi dokumenti, a za 1.0 samo oni koji su najrelevantniji.")
        st.session_state.thold = st.slider(
            'Set relevance (0=any, 1=strict)', 0.0, 1.0, step=0.1, value=0.5)

    # define model, vestorstore and retriever
    llm = ChatOpenAI(model_name=st.session_state.model, temperature=st.session_state.temp,
                     openai_api_key=openai_api_key)
    vectorstore = Pinecone.from_existing_index(
        st.session_state.index_name, embeddings, st.session_state.text, namespace=st.session_state.namespace)
    

    # Prompt template - Loading text from the file
    prompt_file = st.file_uploader(
        "Izaberite početni prompt koji možete editovati ili pišite prompt od početka za definisanje vašeg zahteva", key="upload_prompt", type='txt')
    prompt_t = ""
    if prompt_file is not None:
        prompt_t = prompt_file.getvalue().decode("utf-8")
    else:
        prompt_t = " "
    st.text("")
    # Prompt
    with st.form(key='stilovi', clear_on_submit=False):

        zahtev = st.text_area("Opišite temu, iz oblasti Positive, ili opšte teme. Objasnite i formu željenog teksta: ",
                              prompt_t,
                              key="prompt_prva", height=150)
        submit_button = st.form_submit_button(label='Submit')
        st.session_state.tematika = vectorstore.similarity_search_with_score(zahtev, k=3)
    # pocinje obrada, prvo se pronalazi tematika, zatim stil i na kraju se generise odgovor
    if submit_button:
        with st.spinner("Obrađujem temu..."):
            broj = 1
            doclist=[]
            uk_teme =""
           
            # Iterate through the documents in st.session_state.tematika with enumerate
            for broj, (doc, score) in enumerate(st.session_state.tematika, start=1):
    # Check if the similarity score is greater than st.session_state.thold
                if score > st.session_state.thold:
                    # Append the page content to the selected_docs list
                    doclist.append(doc.page_content)
                    st.info(f"Score sličnosti za dokument broj {broj} je: {round(score, 2)}")
                    # Now, selected_docs contains the page content of documents with a score greater than st.session_state.thold
            uk_teme = doclist
           # ako ne pronadje temu u indexu, trazi na internetu
            if len(doclist) == 0:
                st.info(
                    "Nisam u mogućnosti da pronađem odgovor u indeksu. Pretražujem internet...")
                uk_teme = search.results(zahtev, 4)
            st.info(f"Za relevantnost veću od {st.session_state.thold} broj pronađenih dokumenata je {len(doclist)} ")
            st.info(f"Korišćen je model '{ft_model}' - temperatura je {st.session_state.temp}")
            
          

            # Read prompt template from the file
            sve_zajedno = open_file('prompt_FT.txt')
            system_message_prompt = SystemMessagePromptTemplate.from_template(
                st.session_state.stil)
            system_message = system_message_prompt.format()
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                sve_zajedno)
            human_message = human_message_prompt.format(
                zahtev=zahtev, uk_teme=uk_teme, ft_model=ft_model)
            prompt = ChatPromptTemplate(
                messages=[system_message, human_message])

            # Create LLM chain with chatbot prompt
            chain = LLMChain(llm=llm, prompt=prompt)

            with st.expander("Model i Prompt", expanded=False):
                st.write(f"Korišćen je prompt: {prompt.messages[0].content} ->  {prompt.messages[1].content} - >")
            # Run chain to get chatbot's answer
            with st.spinner("Pišem tekst..."):
                try:
                    st.session_state.odgovor = chain.run(prompt=prompt)
                except Exception as e:
                    st.warning(
                        f"Nisam u mogućnosti da završim tekst. Ovo je opis greške:\n {e}")

    # Izrada verzija tekstova za fajlove formnata po izboru
    # html to docx
    st.text("")
    if st.session_state.odgovor != "":
        with st.expander("FINALNI TEKST", expanded=True):
            st.markdown(st.session_state.odgovor)
        html = markdown.markdown(st.session_state.odgovor)
        buf = html2docx(html, title="Zapisnik")

        x = """
        x = io.BytesIO()
        pisa.CreatePDF(html, dest=x, encoding='utf-8')
        pdf_data = x.getvalue()
        st.download_button(label="Download TekstuStilu.pdf",
                           data=pdf_data,
                           file_name="TekstuStilu.pdf",
                           mime='application/octet-stream')
        """
        st.download_button("Download TekstuStilu.txt",
                           st.session_state.odgovor, file_name="TekstuStilu.txt")
        
        st.download_button(
            label="Download TekstuStilu.docx",
            data=buf.getvalue(),
            file_name="TekstuStilu.docx",
            mime="docx"
        )

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

# Login
st_style()
name, authentication_status, username = positive_login(main, "")
