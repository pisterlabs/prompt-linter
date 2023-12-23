# author: Madhav (https://github.com/madhav-mknc)
# managing the Pinecone vector database

import os 
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain

import pinecone
from langchain.vectorstores import Pinecone


# Initialize pinecone
print("[*] Initializing pinecone...\n")
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"]
)

index_name = os.environ["PINECONE_INDEX_NAME"]
print("[+] Index name:\n",index_name)

NAMESPACE = "madhav"
print("[+] Namespace:",NAMESPACE)

# connecting to the index
index = pinecone.GRPCIndex(index_name)
print(index.describe_index_stats())


# embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# llm
llm = ChatOpenAI(
    model='gpt-4',
# llm = OpenAI(
    temperature=0.3,
    presence_penalty=0.6 
)

# custom prompt
GENIEPROMPT = """
You are an assistant you provide accurate and descriptive answers to user questions, after and only researching through the context provided to you.
You will also use the conversation history provided to you.

Conversation history:
{history}
User:
{question}
Ai: 
"""

prompt_template = PromptTemplate.from_template(GENIEPROMPT)

# chain
chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    verbose=False
)

# for searching relevant docs
docsearch = Pinecone.from_existing_index(
    index_name,
    embeddings
)

# query index
def get_response(query, chat_history=[]):
    docs = docsearch.similarity_search(
        query=query,
        namespace=NAMESPACE
    )

    prompt = {
        "input_documents": docs,
        "question": prompt_template.format(
            question=query, 
            history=chat_history
        )
    }
    
    response = chain(
        prompt,
        return_only_outputs=True)

    return response["output_text"]
