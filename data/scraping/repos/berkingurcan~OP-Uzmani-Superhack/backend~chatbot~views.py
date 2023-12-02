import os
import openai
import tiktoken
import pinecone
import json

from langchain.embeddings.openai import OpenAIEmbeddings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
# Set Pinecone environment. Find next to API key in console
env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

pinecone.init(api_key=api_key, environment=env)

def retrieval(query):
    openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

    embed_model = "text-embedding-ada-002"

    chat = ChatOpenAI(openai_api_key=openai.api_key)

    embed = OpenAIEmbeddings(
        model=embed_model,
        openai_api_key=openai.api_key
    )

    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index('mango')
    vector_store = Pinecone(index, embed.embed_query, "text")

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    xq = openai.Embedding.create(input=query, engine=embed_model)['data'][0]['embedding']
    res = index.query([xq], top_k = 3, include_values=True, include_metadata=True)

    return qa.run(query)


def get_log(request):
    data = {
        'message': 'working'
    }
    return JsonResponse(data)


@csrf_exempt
def get_completion( request, 
                    model="gpt-3.5-turbo",
                    temperature=0, 
                    max_tokens=500):
    if request.method == 'POST':
        # Parse the JSON data from the request body
        data = json.loads(request.body)
        prompt = data.get("messages")
    else:
        return HttpResponse("Invalid request method")
    
    messages = prompt
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return HttpResponse(response.choices[0].message["content"])

@csrf_exempt
def send_query(request, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    if request.method == 'POST':
        # Parse the JSON data from the request body
        data = json.loads(request.body)
        prompt = data.get("messages")
    else:
        return HttpResponse("Invalid request method")
    
    messages = prompt

    embed_model = "text-embedding-ada-002"
    llm = ChatOpenAI(openai_api_key=openai.api_key)

    embed = OpenAIEmbeddings(
        model=embed_model,
        openai_api_key=openai.api_key
    )

    index = pinecone.Index('mango')
    vector_store = Pinecone(index, embed.embed_query, "text")    

    # TODO: set messages to memory
    
    # TODO: Conversational Retrieval Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    query = messages[-1]['content']
    print(query)

    response = qa.run(query)
    return HttpResponse(response)