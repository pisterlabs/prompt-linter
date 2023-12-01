import cohere
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from langchain.embeddings import CohereEmbeddings

from tutor.models import Message
from langchain.document_loaders import PyPDFLoader
from django.core.files.storage import default_storage
import os
from langchain.vectorstores import FAISS

from tutor.models import Page

cohere_key = ""
co = cohere.Client(cohere_key)
embeddings = CohereEmbeddings(cohere_api_key=cohere_key)


def prompt(messages, message):
    prompt_text = f""" `This program designed to be a tutor which will answer questions given to it in short detailed explanations.This program 
    answers questions polite and in detail for the user. The program always provides factual information on the 
    questions asked \n\nQuestion: What is Django and what is it used for?\nAnswer: Django is a Python-based web 
    framework that is used for building web applications quickly and efficiently. It provides a lot of built-in 
    functionality, such as an ORM for database access, a templating engine for rendering views, and a built-in admin 
    interface. Django is used by developers to build web applications of all sizes and complexity 
    levels.\n\nQuestion: What is a REST API and how does it work?\nAnswer:  A REST API is an architectural style for 
    building web APIs that uses HTTP methods (e.g., GET, POST, PUT, DELETE) to manipulate resources over the web. A 
    REST API is stateless, meaning that each request contains all the information necessary for the server to 
    understand and fulfill the request. REST APIs typically return data in JSON or XML format, and are designed to be 
    easily consumed by client applications.\n\nQuestion: What is the difference between a primary key and a foreign 
    key in a relational database?\nAnswer:  A primary key is a unique identifier for a record in a database table, 
    and is used to enforce data integrity and ensure that each record is unique. A foreign key is a field in a table 
    that references the primary key of another table, and is used to establish a relationship between the two tables. 
    Foreign keys are used to enforce referential integrity and maintain consistency between related records in 
    different tables."""

    for text in messages:
        prompt_text += f"\n\nQuestion: {text['question']}"
        prompt_text += f"\nAnswer:{text['answer']}"

    prompt_text += f"\n\nQuestion::{message}?\nAnswer:"

    return prompt_text


def prompt_pdf(page, message):
    prompt_text = "from the following book:"
    for content in page:
        prompt_text += f"""\n from page source{content.page_source}
        \n from page number {content.page_number}
        \n the content: {content.page_content}"""

    prompt_text += f"\n Answer the following question {message}\n Answer:"
    return prompt_text


def home(request):
    return render(request, "chat.html")


@csrf_exempt
def chat(request):
    message = json.loads(request.body)["message"]
    messages = reversed(Message.objects.order_by("-id").values()[:10])
    print(messages)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt(messages, message),
        max_tokens=400,
        temperature=0.9,
        stop_sequences=["\n\nQuestion"])

    answer = response.generations[0].text

    Message(question=message, answer=answer).save()
    return JsonResponse({"message": answer})


def upload_pdf(request):
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('files[]')
        ultimate_pages = []
        for file in uploaded_files:
            path = default_storage.save(f'temp/{file.name}', ContentFile(file.read()))
            loader = PyPDFLoader(f'temp/{file.name}')
            pages = loader.load_and_split()
            ultimate_pages += pages

            os.remove(f'temp/{file.name}')
            print(pages)

        faiss_index = FAISS.from_documents(ultimate_pages, embeddings)
        faiss_index.save_local("index/")
        return redirect("retrieve-from-pdf")

    return render(request, 'upload_pdf.html')


def retrieve_from_pdf(request):
    if request.method == "POST":
        message = request.POST.get("message")
        print(message)
        faiss_index = FAISS.load_local('index/', embeddings)
        docs = faiss_index.similarity_search(message, k=5)

        for i in range(len(docs)):
            docs[i].metadata["page"] = docs[i].metadata["page"] + 1

        return render(request, 'retrieve_from_pdf.html', context={"documents": docs})
    if request.method == "GET":
        return render(request, 'retrieve_from_pdf.html')

    return render(request, 'retrieve_from_pdf.html')

@csrf_exempt
def chat_pdf(request):
    if request.method == "POST" and request.POST.get("form_type") == "send data":
        pages = request.POST.getlist("apage")
        source = request.POST.getlist("asource")
        content = request.POST.getlist("acontent")
        for i in range(len(content)):
            Page(page_source=source[i], page_number=pages[i], page_content=content[i])
        return render(request, 'chat_pdf.html')
    elif request.method == "POST":
        message = json.loads(request.body)["message"]
        page = reversed(Page.objects.order_by("-id").values()[:2])
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=prompt_pdf(page, message),
            max_tokens=400,
            temperature=0.9,
            stop_sequences=["\n\nQuestion"])

        answer = response.generations[0].text

        return JsonResponse({"message": answer})
