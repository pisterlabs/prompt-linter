import os
import PyPDF2
import random
import itertools
import pyttsx3
import streamlit as st
import speech_recognition as sr

from io import StringIO, BytesIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from audio_recorder_streamlit import audio_recorder

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Different GPT Models
MODEL_GPT4_MARCH    = "gpt-4-0314"
MODEL_GPT4          = "gpt-4"
MODEL_GPT3_MARCH    = "gpt-3.5-turbo-16k-0301"
MODEL_GPT3          = "gpt-3.5-turbo-16k"

# Memory
memory = ConversationBufferMemory(memory_key="history", input_key="question")
conversation_history = []

st.set_page_config(page_title="VitroGPT",page_icon=':robot_face:')

def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )

def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: #0e1117;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid white;
                }
            </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_docs(files):
    # st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Por favor ingresa un documento a analizar.', icon="⚠️")
    return all_text




@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    # st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        #query = recognizer.recognize_google(audio, language = "es-MX")
        query = recognizer.recognize_whisper_api(audio)
        print(f"User: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I did not understand what you said.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""
    
def speech_to_text_st(audio_bytes):
    recognizer = sr.Recognizer()

    with sr.AudioFile(BytesIO(audio_bytes)) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

    try:
        #query = recognizer.recognize_google(audio, language = "es-MX")
        query = recognizer.recognize_whisper_api(audio)
        print(f"User: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I did not understand what you said.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def text_to_speech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice", voices[2].id)
    engine.setProperty('rate', 150)  # You can adjust the speech rate (words per minute)
    engine.say(text)
    engine.runAndWait()
    return text

def display_conversation_history(conversation_history):
        # Function to render the chat history in a more interactive way
        for i in range(len(conversation_history)):
            st.markdown(f"**Q{i+1}:** {conversation_history[i][0]}\n **A{i+1}:** {conversation_history[i][1]}")

# ...


def main():
    
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Made by <a href='https://www.vitro.com/es/inicio/'>Vitro</a></p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            
           
       
            
          

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("images/ai_logo.png")


   

   
    with st.container():
        st.write(
            f"""
            <div style="display: flex; align-items: center; margin-left: 0;">
                <h1 style="display: inline-block;">VitroGPT</h1>
                <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
            </div>
            """,
            unsafe_allow_html=True,
        )
        stick_it_good()
    
    


    
    
    st.sidebar.title("Menu")
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])

    
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    uploaded_files = st.sidebar.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:

        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.sidebar.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.sidebar.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings
            # Embed using OpenAI embeddings or HuggingFace embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "HuggingFace Embeddings(slower)":
            # Replace "bert-base-uncased" with the desired HuggingFace model
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)


        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            model=MODEL_GPT4_MARCH, 
            streaming=True, 
            callback_manager=callback_manager, 
            verbose=True, 
            temperature=0
        )

        qa = RetrievalQA.from_chain_type(
            llm=chat_openai, 
            retriever=retriever, 
            chain_type="stuff", 
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "memory": memory
            },
            return_source_documents=True
        )

        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you? Type your question"}]

        with st.container():
            prompt = st.chat_input()
        
        with st.sidebar:
            audio_bytes = audio_recorder(
                text="Listen:    ",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_size="4x",
                pause_threshold=1.0
            )
        
        if st.sidebar.button("📝 Clear Conversation", key='clear_chat_button'):
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you? Type your question"}]
            # move_focus()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if audio_bytes:
            msg = speech_to_text_st(audio_bytes)

        if prompt:
            msg = prompt

        if prompt or audio_bytes:
            st.session_state.messages.append({"role": "user", "content": msg})
            with st.chat_message("Ask your question:"):
                st.write(msg)


        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = qa(msg)
                    answer, docs = res['result'], res['source_documents']
                    st.write(answer) 
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
