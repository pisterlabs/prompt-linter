from django.http import HttpResponse
from django.http import JsonResponse
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

def hello(request):
    return HttpResponse('Hello, Fly!')

def search(request):
    query_string = request.GET.get("query")
    result_list = []

    if query_string is None or query_string == "":
        return JsonResponse(result_list, safe=False)

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "fleek-authority-index"

    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    #query = "Try wearing Allen Edmonds Men's Park Avenue Cap-Toe Oxfords. These black, classic leather shoes are handcrafted and made with high attention to detail. Their sleek, lace-up design adds a formal and quintessential look to any outfit."
    docs = docsearch.similarity_search(query_string, 10)

    for doc in docs:
        result_list.append({"page_content": doc.page_content, "metadata": doc.metadata})

    return JsonResponse(result_list, safe=False)
