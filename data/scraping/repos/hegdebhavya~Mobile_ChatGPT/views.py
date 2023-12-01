from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
import nltk
import magic
import warnings
import json

# Ignore warnings
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "OPEN_API_KEY_HERE"
loader = DirectoryLoader('/home/ubuntu/training_text', glob='**/*.txt')
docs = loader.load()
char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

doc_texts = char_text_splitter.split_documents(docs)
openAI_embeddings = OpenAIEmbeddings(openai_api_key="OPEN_API_KEY_HERE")
vStore = Chroma.from_documents(doc_texts, openAI_embeddings)
model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vStore)
@csrf_exempt
def api_endpoint(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        question = data.get('question')
        print(question)
        response = model.run(question)

        # Return the response as JSON
        return JsonResponse({'response': response})

    # Return an error for unsupported HTTP methods
    return JsonResponse({'error': 'Invalid request method'})


