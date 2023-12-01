from django.shortcuts import render
from .models import *
# Create your views here.

from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def test(request):
    """This function is used for testing the pages"""
    data = [{'heading':f'This is heading_{ind}',
             'content':f'This is content_{ind}'} for ind in range(4)]
    context = {
        'data':data,
    }
    return render(request, 'search.html', context)
#
"""
def home(request):
    print("this is home function executed...")
    return render(request,'search.html')
"""

def search(request):

    query = request.GET['query']

    print(query)

    embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.load_local("/home/kamal/gitfolders/website/vector/vector_db/",embedding)

    docs = db.similarity_search(query,k=7)

    print(docs[0])

    result_docs = []

    for x in docs:
        temp = dict({'source':x.metadata['source'],
                     'content':x.page_content})
        result_docs.append(temp)

    #categs = Category.objects.all()[1:5]

    print(result_docs[0])

    ctx = {
    #    'categories':categs,
        'results':result_docs
    }

    return render(request, 'search.html',ctx)

"""
def reg_search(request):
    query = request.GET['query']
    posts = Post.objects.filter(body__icontains = query)
    print(len(posts))
    print(posts)
    context = {
        'results':posts,
    }
    return render(request,'reg_search.html',context)
"""
