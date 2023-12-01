import sys
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import json


def initialize_qa_chain(data):
    repo_id = "mistralai/Mistral-7B-v0.1"
    llm = HuggingFaceHub(
        huggingfacehub_api_token="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",
        repo_id=repo_id,
        model_kwargs={"temperature": 0.4, "max_new_tokens": 100},
    )
    embeddings = HuggingFaceEmbeddings()
    texts = [entry["transcript"] for entry in data]
    texts = [text.replace("\n", " ") for text in texts]
    db = Chroma.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )


def ask_question(qa_chain, query):
    result = qa_chain({"question": query, "chat_history": []})

    answer = result["answer"].split("\n\n Question: ")[0].strip()

    indexQ = answer.find("\n\nQuestion: ")
    if indexQ != -1:
        answer = answer[:indexQ].strip()

    indexU = answer.find("\n\nUser 1: ")
    if indexU != -1:
        answer = answer[:indexU].strip()

    indexH = answer.find("\n\n## ")
    if indexH != -1:
        answer = answer[:indexH].strip()

    indexR = answer.find("\n\n### Related Questions")
    if indexR != -1:
        answer = answer[:indexR].strip()

    indexA = answer.find("\n\nThis answer is: ")
    if indexA != -1:
        answer = answer[:indexA].strip()

    return answer


if __name__ == "__main__":
    from pathlib import Path

    base_path = Path.cwd()
    data_path = base_path / "data" / "transcriptions.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    qa_chain = initialize_qa_chain(data)

    while True:
        query = input("\nPrompt: ").strip("?")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting")
            sys.exit()
        answer = ask_question(qa_chain, query)
        print("\nAnswer: " + answer + "\n")
