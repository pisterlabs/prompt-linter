from django.shortcuts import render
from .models import *
# Create your views here.

def home(request):
    categs = Category.objects.all() 
    ctx = {
        'categories':categs
    }
    return render(request,'home.html',ctx)

def blog_list(request):
    posts = Post.objects.all()
    categs = Category.objects.all()[:4]
    ctx = {
        'posts':posts,
        'categories':categs
    }
    return render(request,'blog_list.html',ctx)

def blog_detail(request,pk):
    categs = Category.objects.all()
    post = Post.objects.get(pk=pk)
    category = post.category.values()[0]['name']
    posts = Post.objects.filter(category=post.category.values()[0]['id'])[:5]
    #print(category)
    #following will get the posts that are in same category
    ctx = {
        'post':post,
        'categories':categs,
        'category':category,
        'posts':posts
    }
    return render(request, 'blog_detail.html',ctx)

def cat_filter(request,pk):
    categs = Category.objects.all()[1:5]
    posts = Post.objects.filter(category__pk=pk)
    ctx = {
        'cat_id':pk,
        'posts':posts,
        'categories':categs
    }
    return render(request, 'blog_list.html',ctx)

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def search(request):
    query = request.GET['query']
    print(query)

    embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.load_local("/home/kamal/gitfolders/django_projects/website/blogs/yt_data/",embedding)
    
    docs = db.similarity_search(query,k=7)

    result_docs = []
    for x in docs:
        temp = dict({'source':x.metadata['source'],
                     'content':x.page_content})
        result_docs.append(temp)

    categs = Category.objects.all()[1:5]

    ctx = {
        'categories':categs,
        'results':result_docs
    }
    return render(request, 'search.html',ctx)

def reg_search(request):
    query = request.GET['query']
    posts = Post.objects.filter(body__icontains = query)
    print(len(posts))
    context = {
        'results':posts,
    } 
    return render(request,'reg_search.html',context)
