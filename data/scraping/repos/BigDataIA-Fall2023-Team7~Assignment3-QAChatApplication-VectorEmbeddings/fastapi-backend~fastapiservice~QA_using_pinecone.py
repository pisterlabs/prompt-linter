import openai
from typing import List, Iterator
import os
import wget
import pinecone

from ast import literal_eval
from dotenv import load_dotenv
load_dotenv()

# Setting global variables
ENVIRONMENT = os.getenv("ENVIRONMENT")
API_KEY = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPEN_AI_KEY")
index_name = os.getenv("INDEX_NAME")

EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_COMPLETION_MODEL = "gpt-3.5-turbo"


#setting up pinecone database
pinecone.init(api_key=API_KEY,environment=ENVIRONMENT)

# Ignore unclosed SSL socket warnings - optional in case of these errors
import warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

index = pinecone.Index(index_name=index_name)

def get_answer(question, form_titles, top_k=2):

    # Create vector embeddings based on the title column
    ques_embedding = openai.Embedding.create(
                                            input=question,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']

    # Query namespace passed as parameter using title vector
    response = index.query(vector = ques_embedding, 
                                    filter = {"title": {"$in":form_titles}},
                                    top_k=2,
                                    include_metadata=True,
                                    )
    context=''
    if response:
        for i in response['matches']:
            context += i['metadata']['text'] + "\n" 

    message = f'{context} \n\n Question: {question}'

    messages = [
        {"role": "system", "content": "You answer questions using the provided context only"},
        {"role": "user", "content": message},
    ]   

    print(message)

    response = openai.ChatCompletion.create(
        model=CHAT_COMPLETION_MODEL,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message                             

