import streamlit as st
import os
import tempfile
import PyPDF2
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import pyaudio

# Crea un campo di input nella barra laterale dell'interfaccia grafica per inserire la chiave API di OpenAI.
api_key = st.sidebar.text_input("Inserisci la tua OpenaAI API Key 🔑", type="password",
                                placeholder="Inserisci qui la tua OpenAI API Key (sk-...)",
                                help="Puoi ottenere la tua API Key da https://platform.openai.com/account/api-keys.")
# Imposta la chiave API di OpenAI come variabile di ambiente.
os.environ["OPENAI_API_KEY"] = api_key
# Aggiunge una riga orizzontale nella barra laterale dell'interfaccia grafica.
st.sidebar.markdown("---")

# Aggiuge una descrizione nella barra laterale.
st.sidebar.title("About")
st.sidebar.markdown("HAL è un ChatBot vocale basato su Large Language Model, che risponde a domande specifiche sul file PDF che gli viene fornito. Non risponde a domande al di fuori del dominio di conoscenza fornitogli dal file PDF. Inoltre, ha memoria di tutta la conversazione, quindi è in grado di rispondere anche ad una serie di domande collegate fra di loro.")
st.sidebar.markdown("---")
st.sidebar.title("Strumenti utilizzati")        
st.sidebar.markdown('''
                    - [Streamlit](https://streamlit.io/)
                    - [LangChain](https://python.langchain.com/)
                    - [OpenAI](https://platform.openai.com/docs/models)
                     ''')
st.sidebar.markdown("---")
st.sidebar.markdown("Creato da Matteo De Amicis & Gianmarco Venturini 🤝")

# Aggiunge un titolo all'applicazione principale.
st.title("ChatBot HAL 🤖")

# Aggiunge un sottotitolo all'applicazione principale.
st.subheader("Benvenuto! Sono qui per rispondere alle tue domande sul file PDF")

# Crea un campo di caricamento file nella pagina principale dell'interfaccia grafica per caricare un file PDF.
uploaded_file = st.file_uploader("Carica qui il tuo file PDF 📄", type="pdf")
# Verifica se è stato caricato un file PDF.
if uploaded_file is not None:
    # Crea un file temporaneo per salvare il file PDF caricato.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Scrive i dati del file caricato nel file temporaneo.
        tmp_file.write(uploaded_file.read())
        # Ottiene il percorso del file temporaneo.
        tmp_file_path = tmp_file.name

    # Apre il file temporaneo in modalità di lettura binaria.
    with open(tmp_file_path, 'rb') as f:
        # Crea un oggetto PdfReader utilizzando il file PDF aperto.
        pdf_reader = PyPDF2.PdfReader(f)
        # Inizializza una variabile text come stringa vuota
        text = ''
        # Itera su tutte le pagine del file PDF.
        for page in pdf_reader.pages:
            # Estrae il testo da ogni pagina del file PDF e lo aggiunge alla variabile text.
            text += page.extract_text()
    
            
    # Carica il tokenizer GPT-2 preaddestrato.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Definisce una funzione count_tokens che conta il numero di token in un testo.
    def count_tokens(text: str) -> int:
        # Restituisce il numero di token nel testo tokenizzato utilizzando il tokenizer.
        return len(tokenizer.encode(text))

    # Crea un'istanza di RecursiveCharacterTextSplitter con i parametri specificati.
    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens,
    )
    # Suddivide il testo in chunk utilizzando il text_splitter.
    chunks = text_splitter.create_documents([text])

    # Crea un'istanza di OpenAIEmbeddings per calcolare gli embedding del testo.
    embeddings = OpenAIEmbeddings()

    # Crea un database FAISS utilizzando i chunk del testo e gli embeddings.
    db = FAISS.from_documents(chunks, embeddings)

    # Esegue la catena di elaborazione di domanda-risposta utilizzando i documenti di input e la query.
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    query= " "
    docs = db.similarity_search(query)
    chain.run(input_documents=docs, question=query)

    # Crea una catena di elaborazione per il recupero conversazionale utilizzando il modello LLM di OpenAI e il database FAISS come retriever.
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

    # Inizializza una lista chat_history per memorizzare la cronologia delle conversazioni.
    chat_history = []

    # Crea un oggetto Recognizer dal modulo speech_recognition.
    r = sr.Recognizer()

    # Definisce una funzione on_submit che viene chiamata quando viene inviata una domanda.
    def on_submit(query):
        
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))

        st.markdown(f"**Utente:** {query}")
        st.markdown(f"**<span style='color:blue'>HAL</span>:** {result['answer']}", unsafe_allow_html=True)

        risposta_vocale = converti_in_vocale(result['answer'])
        riproduci_audio(risposta_vocale)

        process_audio_input()
        
    # Definisce una funzione recognize_query che converte l'audio in testo utilizzando il riconoscimento vocale.
    def recognize_query(audio):
        try:
            
            text = r.recognize_google(audio, language='it-IT')
            if 'stop' in text.lower():
                st.markdown(f"**<span style='color:blue'>HAL</span>:** Conversazione interrotta. Spero di essere stato d'aiuto!", unsafe_allow_html=True)
                return
            on_submit(text)
        except sr.UnknownValueError:
            st.markdown(f"**HAL:** Non sono riuscito a comprendere l'audio")
            process_audio_input()
        except sr.RequestError as e:
            st.text("Errore durante la richiesta a Google Speech Recognition service: {0}".format(e))
            process_audio_input()
            
    # Definisce una funzione process_audio_input che gestisce l'input audio dell'utente.
    def process_audio_input():
        with sr.Microphone() as source:
            st.markdown(f"**<span style='color:blue'>HAL</span>:** Parla, ti ascolto! Pronuncia **<span style='color:red'>stop</span>** per interrompere la conversazione con HAL.", unsafe_allow_html=True)
            audio = r.listen(source)
            recognize_query(audio)

    # Definisce una funzione converti_in_vocale che converte il testo in un file audio utilizzando il servizio Google Text-to-Speech.
    def converti_in_vocale(testo):
        tts = gTTS(text=testo, lang='it')
        tts.save("risposta_vocale.mp3")
        return "risposta_vocale.mp3"
    
    # Definisce una funzione riproduci_audio che riproduce un file audio.
    def riproduci_audio(file_audio):
        audio = AudioSegment.from_file(file_audio)
        play(audio)

    # Definisce una funzione main che gestisce l'esecuzione principale dell'applicazione.
    def main():
        process_audio_input()

    # Verifica se lo script è eseguito come programma principale.    
    if __name__ == '__main__':
        # Chiama la funzione main per avviare l'applicazione.
        main()


