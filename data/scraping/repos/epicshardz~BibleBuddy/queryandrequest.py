import time
import threading
from dotenv import load_dotenv
from langchain.chains import ChatVectorDBChain
import json
import sys
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import ConversationalRetrievalChain
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
os.environ["QDRANT_HOST"] = os.getenv("QDRANT_HOST")

# Parse the JSON input data
data = json.loads(sys.argv[1])

# Extract the values of the properties
text = data['prompt']
bible = data['selectedOption1']
denom = data['selectedOption2']
modelselection = data['selectedOption3']
last_response = data['last_response']
last_prompt = data['last_prompt']

# print('#########last_response', last_response)
# print('#########last_prompt', last_prompt)

modelname = "gpt-3.5-turbo"
if modelselection == "Slow and quality Answers - GPT-4":
    modelname = "gpt-4"
# print("#######model:", modelselection, 'modelname:', modelname)

system_template = """Use the following pieces of context to try to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer! I repeat, only answer based on the context or to followup on a previous query. If the answer is in the {bible} but not in the context provided please answer using that known scripture.
ALWAYS return a scripture "SOURCES" part in your answer. The scripture "sources" part should be a reference to the verse/verses of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
[line break]
sources: xyz
```

Begin!
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


api_key = os.environ["QDRANT_API_KEY"]
hosturl = os.environ["QDRANT_HOST"]
client = QdrantClient(url=hosturl, prefer_grpc=True, api_key=api_key)
collection_name = bible
embeddings = OpenAIEmbeddings()
vectordb1 = Qdrant(client, collection_name,
                   embedding_function=embeddings.embed_query)
vectortable = vectordb1

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(
    temperature=0, max_tokens=1000, model_name=modelname, max_retries=12, request_timeout=60*3), vectortable.as_retriever(), qa_prompt=prompt, return_source_documents=True)

# chat_history = []
chat_history = [(last_response, last_prompt)]
# chat_history = [(last_prompt, last_response)]
query = text
# result = qa({"question": query, "chat_history": chat_history, })


# threading for stay alive call

# A function to send periodic stay-alive signals

def send_stay_alive():
    stay_alive_signal = '{{{{stay_alive}}}}'
    while not api_call_completed:
        print(stay_alive_signal, flush=True)
        time.sleep(15)  # Send stay-alive signal every 5 seconds


# Wrap the API call in a function
result = ''


def make_api_call():
    global api_call_completed
    global result
    try:
        result = qa(
            {"question": query, "chat_history": chat_history, "bible": bible})
    except Exception as e:
        print("{{{{answer}}}}",
              f"An error occurred during the API call: {e}", "{{{/answer}}}}")
    finally:
        api_call_completed = True
    # Handle the result here


# Set up and start the stay-alive thread
api_call_completed = False
stay_alive_thread = threading.Thread(target=send_stay_alive)
stay_alive_thread.start()

# Call the API
make_api_call()

# Wait for the stay-alive thread to finish
stay_alive_thread.join()

##########


response = result["answer"]

print("{{{{answer}}}}", response, "{{{/answer}}}}")
for doc in result['source_documents']:
    print(
        f"[Document(page_content='{doc.page_content.encode('utf-8')}', lookup_str='', metadata={doc.metadata}, lookup_index=0)]")
