import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()
model = ChatOpenAI()
memory = ConversationBufferMemory(return_messages=True)
embedding_function = OpenAIEmbeddings()


def generate_assistant_response(prompt):
    prompt = (
        """You are a helpful and friendly AI chatbot assistant for Langchain. \n
    Please answer the following user query. You are helping the user learn Langchain,
    so please speak in simple english. Question:\n"""
        + prompt
    )
    db = Chroma(
        persist_directory="./11-Langchain-Bot/langchain_documents_db",
        embedding_function=embedding_function,
    )
    retriever = db.as_retriever(search_type="mmr")

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        # memory=memory,
        return_source_documents=True,
    )
    response = qa(prompt)
    source_string = format_source_string(response)
    full_return = response["result"] + source_string
    return full_return


def format_source_string(response):
    unique_source_documents = set(
        [
            source_document.metadata["source"]
            for source_document in response["source_documents"]
        ]
    )
    source_string = """\n___\n ### Sources: \n ```python
    """
    for source_document in unique_source_documents:
        source_string = (
            source_string
            + source_document
            + """
            """
        )
    return source_string


def save_chat_history(prompt):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt)
    with st.chat_message("assistant"):
        print(assistant_response)
        st.markdown(assistant_response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )


def main():
    st.title("ChatGPT Clone with ConversationChain")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up?")

    if prompt:
        save_chat_history(prompt)


if __name__ == "__main__":
    main()
