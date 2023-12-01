import models
from langchain.chains import ConversationalRetrievalChain
import data

# class QA_Generator:

# def __init__(self, model_id, hf_auth):
#     self.model_id = model_id
#     self.hf_auth = hf_auth
        
def load_chain(llm, courseID, pdf_path = None):    
    # Get vectorstore to use for retrieve
    db = data.embed_and_get_vectorstore(courseID=courseID, pdf_path=pdf_path)

    # Create RetrievalQA
    chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), return_source_documents=True)
    return chain

# def query(self, chain, chat_history=[]):
#     # For just generate questions, don't need to save chat history

#     # query = "Please create questions that follow the same structure as Bloom's Taxonomy for each level."
#     query = "Please create a set of multiple choice questions that follow the Bloom's Taxonomy and also the correct answer for each question."
#     result = chain({"question": query, "chat_history": chat_history})

#     return result, chat_history