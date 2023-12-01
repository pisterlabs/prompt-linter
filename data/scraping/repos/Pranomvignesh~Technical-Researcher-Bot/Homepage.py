# Imports
import PyPDF2
import streamlit as st
import os
import json
import openai
import requests
import tiktoken

from pathlib import Path
from metaphor_python import Metaphor
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Constants
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Technical Researcher Bot",
    page_icon="🤖",
    menu_items={
        'Report a bug': "https://github.com/Pranomvignesh/Technical-Researcher-Bot/issues"
    }
)
CACHE_FOLDER_PATH = Path('./cache').resolve()
FOLDER_PREFIX = 'saved_searches'
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
METAPHOR_API_KEY = st.secrets["METAPHOR_API_KEY"]
OPENAI_MODEL = 'gpt-3.5-turbo'
NO_OF_RESULTS = 1

# Resources


@st.cache_resource
def init_metaphor_instance():
    return Metaphor(METAPHOR_API_KEY)


@st.cache_resource
def init_tokenizer():
    return tiktoken.get_encoding(
        tiktoken.encoding_for_model(OPENAI_MODEL).name
    )

# Methods


def init_metadata(folder_path: str, question: str) -> None:
    meta_data = {
        'folder_path': str(folder_path),
        'question': question
    }
    file_path = folder_path.joinpath('metadata.json')
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(meta_data, indent=4))


def create_folder() -> str:
    num_folders = len(os.listdir(CACHE_FOLDER_PATH))
    new_folder_path = CACHE_FOLDER_PATH.joinpath(
        f'{FOLDER_PREFIX}_{num_folders +1}')
    os.makedirs(new_folder_path)
    return new_folder_path


def init_folder(question: str) -> str:
    new_folder_path = create_folder()
    init_metadata(new_folder_path, question)
    return new_folder_path


def get_topic_from_question(question: str) -> str:
    # openai.api_key = OPENAI_API_KEY
    SYSTEM_MESSAGE = """
    You are helpful assistant. You have to summarize the QUESTION asked by the user into a 
    TOPIC which can be used to retrieve technical papers.
    
    EXAMPLES:
    ---
    QUESTION : "Give me information about tracking objects in real time camera feed"
    TOPIC : "Here is a paper about : Object Tracking in real time camera feed"
    ---
    QUESTION : "Best strategies to preprocess the natural language text"
    TOPIC : "Here is a paper about : Strategies for Preprocessing Natural Language Text"
    ---
    
    You have to output only the topic. Don't output anything else
    
    You should not output like this
    TOPIC : "Here is a paper about : Strategies for Preprocessing Natural Language Text"
    You have to output like this
    Here is a paper about : Strategies for Preprocessing Natural Language Text
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content


def get_urls_using_metaphor(question, num_results) -> None:
    metaphor = init_metaphor_instance()
    topic = get_topic_from_question(question)
    search_response = metaphor.search(
        topic,
        use_autoprompt=True,
        num_results=num_results or NO_OF_RESULTS,
        include_domains=['https://arxiv.org/']
    )
    return [result.url for result in search_response.results]


def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    else:
        print(f"Failed to download PDF from {url}")
        return False


def convert_url_to_pdf_url(url: str) -> str:
    paper_id = urlparse(url).path.split('/')[-1]
    if paper_id.endswith('.pdf'):
        paper_id = paper_id.replace('.pdf', '')
    return f'https://arxiv.org/pdf/{paper_id}.pdf'


def save_pdf(url: str, temp_path: str) -> bool:
    print("Url:\n", url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(temp_path, 'wb') as file:
            file.write(response.content)
        return True
    return False


def extract_contents(urls: list[str], new_folder_path: Path) -> list[str]:
    temp_path = str(new_folder_path.joinpath('temp.pdf'))
    contents = []
    for url in urls:
        is_pdf_saved = save_pdf(url, temp_path)
        if is_pdf_saved:
            text = ""
            with open(temp_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            contents.append(text)
            os.remove(temp_path)
    print("Contents:\n", contents)
    return contents


def custom_length_function(text):
    tokenizer = init_tokenizer()
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def create_chunks(contents: list, new_folder_path: str) -> list:
    # data_path = new_folder_path.joinpath('data.jsonl')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=custom_length_function,
        separators=['\n\n', '\n', ' ', '']
    )
    all_chunks = []
    for content in contents:
        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
    return all_chunks


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


def create_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0, streaming=True),
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def init_chatbot(question: str, num_results: str) -> None:
    num_results = int(num_results)
    new_folder_path = init_folder(question)
    try:
        with st.spinner("Fetching relevant research papers..."):
            urls = get_urls_using_metaphor(question, num_results)
            pdf_urls = [convert_url_to_pdf_url(url) for url in urls]
        with st.spinner("Extracting Content from the research papers..."):
            list_of_contents = extract_contents(pdf_urls, new_folder_path)
        with st.spinner("Preprocessing the Content..."):
            chunks = create_chunks(list_of_contents, new_folder_path)
        with st.spinner("Initializing the Chat Bot..."):
            vectorstore = create_vectorstore(chunks)
            st.session_state.conversation_chain = create_conversation_chain(
                vectorstore)
    except Exception as e:
        st.write(f'{e.__class__.__name__} : {str(e)}')

# Main Page UI


def main():
    st.title("""Technical Researcher Bot""")

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = OPENAI_MODEL

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "question" not in st.session_state:
        st.session_state.question = ''

    if "show_chatbot" not in st.session_state:
        st.session_state.show_chatbot = False

    # Use with sidebar here
    with st.sidebar:
        with st.form("my_form"):
            question = st.text_input(
                'Enter the topic you want to research about',
                key='question'
            )
            num_results = st.selectbox(
                'Number of papers to retrieve',
                (1, 2, 3, 5, 10),
                key='num_results'
            )
            submitted = st.form_submit_button("Create Chatbot")
            if submitted:
                init_chatbot(question, num_results)
                st.session_state.messages = []
                st.session_state.show_chatbot = True

    if st.session_state.show_chatbot:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if query := st.chat_input("Hi! I'm your AI Research Assistant. What can I help you with?"):
            st.session_state.messages.append(
                {"role": "user", "content": query}
            )
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                messages.append({"role": "user", "content": query})
                response = st.session_state.conversation_chain({
                    'question': query
                })
                st.session_state.chat_history = response['chat_history']
                print(response)
                full_response = response['chat_history'][-1].content
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.write("Enter your question in the sidebar to initialize the chat bot")

        st.markdown("## Video Demo - 1")

        video_file = open(
            'assets/video-demos/streamlit-Homepage-2023-10-02-15-10-95.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        st.markdown("## Video Demo - 2")

        video_file = open(
            'assets/video-demos/streamlit-Homepage-2023-10-02-16-10-66.webm', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)


if __name__ == "__main__":
    main()
