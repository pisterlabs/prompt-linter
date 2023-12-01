import pdf2image
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

if "conversational_retriever_chain" not in st.session_state:
    st.session_state.conversational_retriever_chain = None

if "talk_with_pdfs_memory" not in st.session_state:
    st.session_state.talk_with_pdfs_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "translator" not in st.session_state:
    st.session_state.translator = False

if "talk_with_pdfs_files" not in st.session_state:
    st.session_state.talk_with_pdfs_files = None


def read_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        for page in pdf2image.convert_from_bytes(uploaded_file.read()):
            text += pytesseract.image_to_string(page)

    return text


def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
    )
    memory = st.session_state.talk_with_pdfs_memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return chain


def handle_human_message(user_input):
    def translate(text):
        # Translate user input to English
        print("Translating to English...")
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
        )

        messages = [
            SystemMessage(content="You are a translator to English. You only say Human's messages in English."),
            HumanMessage(content=text),
        ]

        return llm(messages)

    if st.session_state.translator:
        translated = translate(user_input)
    else:
        translated = HumanMessage(content=user_input)

    with st.spinner(text="Searching in your documents..."):
        st.session_state.conversational_retriever_chain(
            {'question': translated.content}
        )


def main():
    # set page config
    st.set_page_config(
        page_title="Chat with PDFs 📚",
        page_icon="📚",
        initial_sidebar_state="expanded",
    )

    # Add title and subtitle
    st.title(":orange[bit AI] 🤖")
    st.caption(
        "bitAI powered by these AI tools:"
        "OpenAI GPT-3.5-Turbo 🤖, HuggingFace 🤗, CodeLLaMa 🦙, Replicate and Streamlit of course."
    )
    st.subheader("Chat with PDFs 📚")

    col1, col2 = st.container().columns([1, 3])
    with col1:
        translator = st.checkbox("Translate to English")
        if translator:
            st.session_state.translator = True
        else:
            st.session_state.translator = False

    with col2:
        st.caption("Use the checkbox to translate your messages to English.")

    with st.sidebar:
        st.session_state.talk_with_pdfs_files = st.file_uploader(
            label="Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )
        upload_button = st.button("Upload PDFs", disabled=not st.session_state.talk_with_pdfs_files)
        if upload_button:
            with st.spinner(text="Processing PDFs..."):
                raw_text = read_pdfs(st.session_state.talk_with_pdfs_files)
                text_chunks = get_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversational_retriever_chain = get_conversation_chain(vector_store)

        clear_button = st.button("Clear History")
        if clear_button:
            st.session_state.talk_with_pdfs_memory.clear()

    prompt = st.chat_input("Ask me anything about the PDFs you uploaded.")
    if prompt:
        handle_human_message(prompt)

    for message in st.session_state.talk_with_pdfs_memory.buffer_as_messages:
        if isinstance(message, HumanMessage):
            st.chat_message("Human", avatar="👤").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("AI", avatar="🤖").write(message.content)


if __name__ == "__main__":
    main()
