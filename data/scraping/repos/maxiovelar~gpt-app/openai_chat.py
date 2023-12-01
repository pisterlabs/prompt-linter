import json
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from api.pinecone_db import get_db_instance

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


################################################################################
# Make OpenAI chat query
################################################################################
def query_chat(query):
    db = get_db_instance()
    prompt_template = """
    #You are a shopping assistant. Use the following pieces of context to answer the question at the end. Take your time to think and analyze your answer. Just answer the user question if is related with products, if you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    #Return a conversational answer about the question without any id in a 'text' key.
    #Return an array with product id's in a 'products' key just if you found products for the user question.
    
    #Don't return duplicated products.
    #Don't return non active products.
    #Don't show products that don't exist in the database.
    

    #Context: {context}
    #Question: {question}
    #Answer in JSON format:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0,openai_api_key=os.getenv('OPENAI_API_KEY')) 
    retriever=db.as_retriever()
    
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )
    
    res = qa.run(query)

    return json.loads(res)
