import json
from pathlib import Path
from typing import Union

from langchain import Cohere, HuggingFaceHub, Petals
from langchain.chains import ChatVectorDBChain
from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

# We'll save our headlines to this path
dataset_path = "./researcher/data/dataset.json"
file_path = "./researcher/data/abstracts.txt"

import hashlib

from dotenv import load_dotenv


def get_abstracts():
    dataset = json.load(open(dataset_path))

    papers = []
    for paper in dataset:
        title = paper["title"]
        abstract = paper["abstract"]
        paper_info = f"{title}: {abstract}"
        papers.append(paper_info)

    content = "\n\n".join(papers)
    file_hash = hashlib.sha256(content.encode()).hexdigest()

    with open(f"{file_path}_{file_hash}", "w") as f:
        f.write(content)

    print("Total of {} abstracts".format(len(papers)))
    return file_hash


def get_full_papers():
    raise NotImplementedError("This is not implemented yet")


def load_llm(llm_model="openai-gpt", temp=0.0, load_qa=False, verbose=False):
    # default is OpenAI GPT
    if llm_model == "openai-gpt" or llm_model is None:
        llm = OpenAI(temperature=temp, max_tokens=1024)

    elif llm_model == "cohere":
        llm = Cohere(
            temperature=temp,
            model="command-xlarge-nightly",
            max_tokens=1024,
            truncate="END",
        )

    elif llm_model in [
        "google/flan-t5-xxl",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-j-6B",
        "EleutherAI/pythia-12b-deduped",
    ]:
        llm = HuggingFaceHub(
            repo_id=llm_model,
            model_kwargs={
                "temperature": temp,
                # "truncation": "only_first"
            },
        )
    elif llm_model == "bigscience/petals":
        llm = Petals(
            model_name=llm_model,
            temperature=temp,
        )

    # elif llm_model == "kworts/BARTxiv":
    #     llm = HuggingFaceHub(repo_id=llm_model, model_kwargs={"temperature": temp})

    else:
        raise ValueError(f"Unknown LLM model: {llm_model}")

    return load_qa_chain(llm, chain_type="stuff", verbose=verbose) if load_qa else llm


def load_vectorstore(source="abstract"):
    if source == "abstract":
        file_hash = get_abstracts()
    elif source == "paper":
        get_full_papers()
    else:
        raise ValueError(f"Unknown source: {source}")

    # check if we already have the vectorstore saved
    # this is done by hardcoding hashses in a file
    load_dotenv()
    if file_hash in open("./vectorstore/hashes").read():
        return FAISS.load_local(f"./vectorstore/{file_hash}", OpenAIEmbeddings())

    with open(f"{file_path}_{file_hash}") as f:
        saved_file = f.read()
        # Split the text to conform to maximum number of tokens
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1024,
            chunk_overlap=256,
            # length_function=len,
        )

        texts = text_splitter.split_text(saved_file)

        vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
        Path(f"./vectorstore/{file_hash}").mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(f"./vectorstore/{file_hash}")
        with open("./vectorstore/hashes", "a") as f:
            f.write(file_hash + "\n")

    return vectorstore


def load_chat_chain(llm, vectorstore):
    chain = ChatVectorDBChain.from_llm(
        llm, vectorstore, top_k_docs_for_context=4, return_source_documets=False
    )
    return chain


# Answer questions about the papers
def start_conversation(
    query,
    chat_history=[],
    vectorstore=None,
    model: Union[str, LLM, Chain] = None,
    temp=0.0,
):
    # Index the papers
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # load llm if model is a string
    if model is None:
        model = "openai-gpt"

    if isinstance(model, str):
        llm = load_llm(model, temp=temp, load_qa=False, verbose=False)
        chain = chain = load_chat_chain(llm, vectorstore)
    elif isinstance(model, LLM):
        chain = load_chat_chain(model, vectorstore)
    elif isinstance(model, Chain):
        chain = model
    else:
        raise ValueError(f"Unknown model: {model}")

    res = chain({"question": query, "chat_history": chat_history})
    return res


if __name__ == "__main__":
    import sys

    import dotenv

    dotenv.load_dotenv(".env")

    # Receive query from stdin
    chat_history = []
    faiss_store = load_vectorstore()
    while True:
        print("Enter your query: ")
        query = sys.stdin.readline()

        if not query:
            break

        out = start_conversation(
            query=query, chat_history=chat_history, vectorstore=faiss_store
        )
        # update chat history
        chat_history.append((query, out["answer"]))

        print("Answer: ")
        print(out["chat_history"])
        print(out["answer"])

        # query = "Create a list of the main techniques used. Provide a brief description of each technique."
