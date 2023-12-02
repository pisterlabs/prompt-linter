import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone 
from langchain.memory import ConversationBufferMemory

load_dotenv()
openai_key = os.environ.get('OPENAI_API_KEY')
# pinecone_key = os.environ.get('PINECONE_API_KEY')
# pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index = "langchain1"


app = FastAPI(
    title="LangChain DocsGPT",
    description="The backend for LangChain DocsGPT.",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_document(message): # There may be a better/default way to do this but not sure; this specific chain doesn't seem to have built in memory.
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    return Document(page_content=message, metadata={})

def answer_question(question: str, vs, chain, memory):
    query = question
    docs = vs.similarity_search(query)
    conversation_history = memory.load_memory_variables(inputs={})["history"]
    context_window = conversation_history.split("\n")[-3:] 
    conversation_document = convert_to_document(context_window)
    input_documents = docs + [conversation_document]

    answer = chain.run(input_documents=input_documents, question=query)
    memory.save_context(inputs={"question": question}, outputs={"answer": answer})
    docs_metadata = []
    for doc in docs:
        metadata = doc.metadata
        if metadata is not None:
            doc_metadata = {
                "title": metadata.get('title', None),
                "relURI": metadata.get('relURI', None)
            }
            docs_metadata.append(doc_metadata)

    return {"answer": answer, "docs": docs_metadata}


llm = OpenAI(temperature=0, openai_api_key=openai_key, max_tokens=-1) 
chain = load_qa_chain(llm, chain_type="stuff")
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
docsearch = Pinecone.from_existing_index(pinecone_index, embeddings)
memory = ConversationBufferMemory()

def prompt_question(memory):
    while True:
        question = input("What is your question? (Type 'exit' to quit) ")
        if question.lower() == 'exit':
            break
        print(answer_question(question=question, vs=docsearch, chain=chain, memory=memory))
        print("\n")
    conversation_history = memory.load_memory_variables(inputs={})["history"]
    print("Conversation History:")
    print(conversation_history)

if __name__ == "__main__":
    memory = ConversationBufferMemory()
    prompt_question(memory)