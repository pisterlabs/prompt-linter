# Importing required libraries and modules
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
import os
import logging


def setup_logging():
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)


setup_logging()

# Configuring the Streamlit page appearance
st.set_page_config(
    page_title="Ask Anything AI Chat Bot", page_icon="🤖", layout="wide"
)


def settings():
    # Importing necessary modules for creating vector stores and embeddings
    import faiss
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore

    # Initializing the OpenAI embeddings model
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536

    # Creating a FAISS index for storing embeddings
    index = faiss.IndexFlatL2(embedding_size)

    # Creating a FAISS vector store using the embeddings and index
    vectorstore_public = FAISS(
        embeddings_model.embed_query, index, InMemoryDocstore({}), {}
    )

    # Initializing the language model (GPT-4 in this case)
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-4")

    # Setting up a Google Search API Wrapper for web retrieval
    from langchain.utilities import GoogleSearchAPIWrapper

    search = GoogleSearchAPIWrapper()

    # Initializing the Web Research Retriever with necessary components
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=3
    )

    return web_retriever, llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        # Initializing the StreamHandler with a container to display text and initial text
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Updating the displayed text as new tokens are generated by the language model
        self.text += token
        self.container.markdown(self.text + "▌")


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        # Initializing the PrintRetrievalHandler with a container to display retrieved documents
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        # Displaying the question/query when the retrieval starts
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # Displaying the retrieved documents when the retrieval ends
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


# Displaying the main header and information about the application
st.header("Ask Anything AI Chat Bot")
st.info(
    "I can help answer any questions by searching Google"
)

# Initializing the retriever and language model if they haven't been initialized yet
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# Initializing a list to store messages if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying all the previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Input field for the user to ask a question
if question := st.chat_input("Ask a question:"):
    try:
        # Storing the user's question and displaying it in the chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Display loading wheel while processing the question
        with st.spinner("Processing your question..."):
            # Initializing the QA chain with the language model and the web retriever
            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm, retriever=web_retriever
            )

            # Setting up callback handlers to manage the display of retrieval results and the generated answer
            retrieval_streamer_cb = PrintRetrievalHandler(st.container())
            answer = st.empty()
            stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")

            # Executing the QA chain to generate and display the answer
            result = qa_chain(
                {"question": question},
                callbacks=[retrieval_streamer_cb, stream_handler],
            )

        # Storing the full response and displaying it in the chat
        full_response = "`Answer:`\n\n" + result["answer"]
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        with st.chat_message("assistant"):
            st.markdown(full_response)

        # Displaying the sources of the information provided in the answer
        st.info("`Sources:`\n\n" + result["sources"])

    except Exception as e:
        st.error(
            "Sorry, an error occurred while processing your question. Please try again later."
        )
        logging.error("Error processing question: %s", str(e))