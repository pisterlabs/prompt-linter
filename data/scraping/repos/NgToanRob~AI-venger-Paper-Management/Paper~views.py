from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from .utils import get_text, summarize_and_create_vectordb, get_llm
import json
import os


messages = [
    (
        "Bot",
        "👋 Hello there! As your personal document assistant, I'm here to assist you with any document-related needs you may have. How can I help you today?",
    ),
]


# Create your views here.
@csrf_exempt
def chatpaper(request, id):
    url = "https://arxiv.org/pdf/" + id

    data = json.loads(request.body)

    query = data["query"]

    if query == "":
        text = get_text(url)
        summary = summarize_and_create_vectordb(text)
        return JsonResponse({"summary": summary})

    embeddings = HuggingFaceEmbeddings()
    # Load the vector store for similarity search
    new_db = FAISS.load_local("faiss_index", embeddings)

    # Perform similarity search on the query
    docs = new_db.similarity_search(query, k=2)

    # Get the language model for question answering
    llm = get_llm()

    # Load the question answering chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Perform question answering on the input documents
    answer = chain.run(input_documents=docs, question=query)

    return JsonResponse({"answer": answer})
