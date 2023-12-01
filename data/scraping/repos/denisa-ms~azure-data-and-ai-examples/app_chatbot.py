from dotenv import load_dotenv
import os
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME")



def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])


def ask_question_with_context(qa, question, chat_history):
    query = "what is Azure OpenAI Service?"
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history




if __name__ == "__main__":
    # Configure OpenAI API
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_type="azure")
    
    embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)


    # Initialize gpt-35-turbo and our embedding model
    #load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    #use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)


    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)