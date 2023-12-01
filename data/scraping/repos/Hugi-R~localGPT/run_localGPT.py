import logging
import time

import click
import langchain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME, 
    PERSIST_DIRECTORY, 
    MODEL_PATH,
    MODEL_MAX_CTX_SIZE,
    MODEL_STOP_SEQUENCE, 
    MODEL_PROMPT_TEMPLATE, 
    MODEL_GPU_LAYERS, 
    MODEL_TEMPERATURE,
    MODEL_PREFIX
)

def load_model(lc_callback: BaseCallbackHandler = None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Returns:
        LlamaCpp: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model is provided.
    """
    logging.info(f"Loading Model: {MODEL_PATH}, with {MODEL_GPU_LAYERS} gpu layers")
    logging.info("This action can take some time!")

    if MODEL_PATH is not None:
        max_ctx_size = MODEL_MAX_CTX_SIZE
        max_token = min(512, MODEL_MAX_CTX_SIZE) # we limit the amount of generated token. We want the answer to be short.
        kwargs = {
            "model_path": MODEL_PATH,
            "n_ctx": max_ctx_size,
            "max_tokens": max_token, 
        }
        kwargs["n_gpu_layers"] = MODEL_GPU_LAYERS
        kwargs["n_threads"] = 8
        kwargs["stop"] = MODEL_STOP_SEQUENCE
        kwargs["n_batch"] = 512 # faster prompt evaluation. It's important to speed it up because it contain the context
        kwargs["callbacks"] = [lc_callback]
        kwargs["temperature"] = MODEL_TEMPERATURE # default is 0.8, values between 0 and 1 doesn't affect much the result
        return LlamaCpp(**kwargs)

    raise ValueError("No model provided")

def setup_qa(lc_callback: BaseCallbackHandler = None):
    if lc_callback is None:
        lc_callback = StreamingStdOutCallbackHandler()

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder="models/.cache",
        model_kwargs={"device": "cpu"} # Using cpu for embeddings does not appear to impact the user experience, but it free up VRAM for the LLM, wich is a big win.
    )

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=MODEL_PROMPT_TEMPLATE)
    memory = ConversationBufferMemory(
        input_key="question",
        memory_key="history",
        human_prefix=MODEL_PREFIX["human"],
        ai_prefix=MODEL_PREFIX["ai"],
    )

    llm = load_model(lc_callback)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return qa

# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Show langcahin debug (Default is False)",
)
def main(show_sources, debug):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, is HuggingFaceBgeEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    langchain.debug=debug

    logging.info(f"Display Source Documents set to: {show_sources}")

    qa = setup_qa()

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        t_start = time.time()
        res = qa(query)
        docs = res["source_documents"]
        t_end = time.time()

        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"])
                print(document.page_content)
        else:
            sources = set([d.metadata["source"] for d in docs])
            for s in sources:
                print(f"> {s}")
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

        print(f"Done in {t_end - t_start:0.1f}s")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.ERROR
    )
    main()
