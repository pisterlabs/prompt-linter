
from util import StreamHandler, PrintRetrievalHandler, get_parent_dir
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
import os
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import torch
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(
    filename='document_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def initialize_app():
    # Initialize the app, including model setup, retriever, and other initializations
    global llm, qa_chain

    # Initialization code for model and retriever

    model_name = os.environ.get("EMBEDING_MODEL", "thenlper/gte-base")
    # set True to compute cosine similarity
    encode_kwargs = {'normalize_embeddings': True}

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    # Here is the new embeddings being used
    embedding = model_norm

    parent_directory = get_parent_dir()
    persist_directory = os.path.join(os.getcwd(), 'db')

    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    retriever_k = int(os.environ.get("RETRIEVER_K","5"))
    retriever = vectordb.as_retriever(search_kwargs={"k": retriever_k})
    

    # Initialization code for Streamlit sidebar and default message
    # ...

    with st.sidebar:
        option = st.selectbox(
            "Local LLM or OPENAI API?",
            ("OPENAI", "Local LLM"),
        )

        model_dir = os.path.join(os.getcwd(), "models")

        model_file_bin = []

        for filename in os.listdir(model_dir):
            if filename.endswith(".bin") or filename.endswith(".gguf"):
                model_file_bin.append(filename)

        if (option == "Local LLM"):
            n_ctx = int(os.environ.get("N_CTX","2048"))
            n_gpu_layers = int(os.environ.get("N_GPU_LAYERS","8"))
            n_batch = int(os.environ.get("N_BATCH","100"))

            try:
                local_llm_file = st.selectbox(
                    "Llama GGUF file", model_file_bin)
                model_path = os.path.join(model_dir, local_llm_file)

                llm = LlamaCpp(model_path=model_path, temperature=0, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                               n_batch=n_batch, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
                st.success(f"Local LLM: {local_llm_file} has been loaded")
                logging.info(f"Local LLM: {local_llm_file} has been loaded")
            except:
                st.error(f"Failed to load model, Please select a valid gguf model")
                logging.error(f"Failed to load model, Please select a valid gguf model")
                exit(1)

        else:

            if load_dotenv():
                st.success(".env has been found")
                logging.info(".env has been found")

            try:
                llm = ChatOpenAI(streaming=True, temperature=0.0, callbacks=[
                                 StreamingStdOutCallbackHandler()])
            except:
                st.error(
                    "Please enter a valid OPENAI api key in dotenv or use Local LLM.")
                logging.error(
                    "Please enter a valid OPENAI api key in dotenv or use Local LLM.")
                exit(1)

    template = """
    You are a helpful, respectful, and honest assistant dedicated to providing informative and accurate response based on provided context((delimited by <ctx></ctx>)) only. You don't derive
    answer outside context, while answering your answer should be precise, accurate, clear and should not be verbose and only contain answer. In context you will have texts which is unrelated to question,
    please ignore that context only answer from the related context only.
    If the question is unclear, incoherent, or lacks factual basis, please clarify the issue rather than generating inaccurate information.

    If formatting, such as bullet points, numbered lists, tables, or code blocks, is necessary for a comprehensive response, please apply the appropriate formatting.

    <ctx>
    CONTEXT:
    {context}
    </ctx>

    QUESTION:
    {question}

    ANSWER
    """


    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)



    chain_type_kwargs={
        "verbose": True,
        "prompt": QA_CHAIN_PROMPT,

    }
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=False, verbose=True,
                                           chain_type_kwargs=chain_type_kwargs)


def process_user_message(prompt):
    # Process a user message and return the assistant's response
    global qa_chain
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.spinner("Processing..."):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())

        response = qa_chain.run(
            prompt, callbacks=[retrieval_handler, stream_handler])

        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response))
        # logging.info(f"User prompt: {prompt}")
        # logging.info(f"Assistant response: {response}")


def main():
    st.title("InfoGPT")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(
            role="assistant", content="How can I help you?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        process_user_message(prompt)


if __name__ == "__main__":
    load_dotenv()
    initialize_app()
    main()
