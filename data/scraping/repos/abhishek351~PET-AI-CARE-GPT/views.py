from django.shortcuts import render
from langchain.llms import CTransformers
from langchain.llms import OpenAI
# Create your views here.
# myapp/views.py
from django.http import JsonResponse
from langchain.chains import RetrievalQA    
from rest_framework.response import Response

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from rest_framework.decorators import api_view
from langchain import PromptTemplate

import os

os.environ["OPENAI_API_KEY"] = ""
embeddings =  OpenAIEmbeddings()


llm=OpenAI()

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# View for creating embeddings
@api_view(["POST"])
def create_embeddings(request):
    if request.method == 'POST':
    


        pdf_directory = 'E:\Dog AI care\myproject\myapp\pdf'
        loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)
            
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")

        response = {
            "message": "Vector Store Created",
        }

        return JsonResponse(response)

# View for querying
@api_view(["POST"])
def query_embeddings(request):
    if request.method == 'POST':
        # Your code for querying here

        # Example code for querying
        query = request.data.get("question")
        print(query)
        
        



        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        chain_type_kwargs = {"prompt": prompt}
        load_vector_store = Chroma(persist_directory="stores\pet_cosine", embedding_function=embeddings)
        retriever = load_vector_store.as_retriever(search_kwargs={"k":1})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents = True,
            chain_type_kwargs= chain_type_kwargs,
            verbose=True
        )
        # Perform querying here and prepare the response
        res=qa(query)
        print(res)

        return Response({
            "question": query,
            "answer": res["result"],
            
            
        })

        # return render(request, 'qa_interface.html', context)
