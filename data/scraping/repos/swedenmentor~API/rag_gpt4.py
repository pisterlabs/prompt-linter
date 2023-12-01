from langchain.llms import OpenAI, HuggingFacePipeline
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI # if you want to use GPT-4
import os

load_dotenv()
#%% 1. Initializing the LLM Model (GPT-4)


# Need to change respective OPENAI key in .env file
llm = AzureChatOpenAI(deployment_name="gpt4",
                      model_name="gpt-4")


#%% 2. Building the Knowledge Base
#######################
# Step 2: Vector DB containing the specialized data
#######################

import pinecone
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)
index_name = 'llama-2-rag'
index = pinecone.Index(index_name)

# For check properties of vector DB
#index.describe_index_stats()
#(it seems we load data directly from pinecone without indexing)


#%% 3. Initializing the Embedding Pipeline (text-embedding-ada-002)
#######################
# Embed model
# maps sentences & paragraphs to a 384-dimensional dense vector space
# and can be used for tasks like clustering or semantic search.
#######################
from langchain.embeddings import OpenAIEmbeddings

embed_model_id = 'text-embedding-ada-002'
embed_model = OpenAIEmbeddings(
    deployment='embedding',
    model=embed_model_id
)

#%% 4. Initializing the RetrievalQA Component

#######################
# Step 3: Langchain to glue them together
#######################

from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

text_field = 'text'  # field in metadata that contains text content
vectorstore = Pinecone(index,
                       embed_model,
                       text_field)
generate_text = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=vectorstore.as_retriever())

#######################
# Result
#######################
# print(generate_text("What is quantum Physics?"))
# {'query': 'What is quantum Physics?', 'result': " I don't know."}

def text_transform(res):
    source_documents = []
    for document in res['source_documents']:
        doc = document.to_json()
        source_documents.append(doc['kwargs']['metadata'])
    return json.dumps({
        'result': res['result'],
        'source_documents': source_documents
    })

def build_prompt(messages):
    return messages[-1]['content']

#print(text_transform(generate_text("What is deep convolutional nets?")))
