from django.shortcuts import render
from .models import Lawyer
from .filters import LawyerFilter
from django.core.paginator import Paginator
from .forms import RecommenderForm 
from langchain.llms import OpenAI
import pandas as pd


from qdrant_client import QdrantClient
from qdrant_client.models import models
from sentence_transformers import SentenceTransformer

from .utils import load_data, prepare_data
import pickle, os

def load_vectors(vector_file):
    with open(os.path.join(vector_file),"rb") as f:
        my_object = pickle.load(f)
    return my_object

# Create your views here.

# Create a VectorDB client

client = QdrantClient(":memory:")
client.recreate_collection(collection_name="lawyers_collection",
                           vectors_config=models.VectorParams(
                               size=384, distance=models.Distance.COSINE
                           ))

# Vectorized our Data: Create Word Embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')                                                                                                                                                                     
# model = OpenAI(openai_api_key="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")

df = load_data(r"C:\Users\a21ma\OneDrive\Desktop\Datahack\DataHack_2_Tensionflow\Vector Database\recproject\FINALFINALFINALdataset.csv")
docx, payload = prepare_data(df)
vectors = model.encode(docx, show_progress_bar=True)
# vectors = load_vectors(r"C:\Users\a21ma\OneDrive\Desktop\Datahack\DataHack_2_Tensionflow\Vector Database\recproject\recommender\vectorized_lawyers.pickle")
# Store in VectorDB collection

client.upload_collection(collection_name="lawyers_collection",
                            payload=payload,
                            vectors=vectors,
                            ids=None,
                            batch_size=256)



def index_view(request):
    lawyers = Lawyer.objects.all()
    search_filter = LawyerFilter(request.GET, queryset=lawyers)
    lawyers = search_filter.qs
    paginator = Paginator(lawyers, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"title":"Recommender App", "lawyers": lawyers,
               "search_filter": search_filter,
               "page_obj": page_obj,
               }
    return render(request, "index.html", context)

# def read_lawyers(request,pk):
#     lawyer = Lawyer.objects.get(pk=pk)
#     context = {"lawyer":lawyer}
#     return render(request, "read_lawyers.html", context)

def recommend_view(request):
    if request.method == "POST":
        form = RecommenderForm(request.POST)
        if form.is_valid():
            search_term = form.cleaned_data["search_term"]
            # vectorized the search term
            vectorized_text = model.encode(search_term).tolist()
            results = client.search(collection_name="lawyers_collection",
                                    query_vector=vectorized_text,
                                    limit=10),
        
            # search the VectDB and get recommendation
            context = {"results": results,"form":form,"search_term":search_term} 
            return render(request, "recommend.html", context)
    else:
        form = RecommenderForm()
    context = {"form": form}
    return render(request, "recommend.html", context)

# def integration_view(request):
#     form = RecommenderForm() 
#     if request.method == "POST":
#         search_term = request.POST.get("search")
#             # vectorized the search term
#         vectorized_text = model.encode(search_term).tolist()
#         results = client.search(collection_name="lawyers_collection",
#                                 query_vector=vectorized_text,
#                                 limit=5),
#         # search the VectDB and get recommendation
#         context = {"results": results,"form":form,"search_term":search_term} 
#         return render(request, "recommend.html", context)