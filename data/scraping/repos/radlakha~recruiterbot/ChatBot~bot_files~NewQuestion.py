import os

from langchain import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
# from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from bot_files.config import setup_key
setup_key() # OpenAI key

# Load saved embeddings index
docsearch = FAISS.load_local("faiss_index", OpenAIEmbeddings())

llm_gen = OpenAI(temperature=0,verbose=True) #Instantiate an LLM 

question_generator = LLMChain(llm=llm_gen, prompt=CONDENSE_QUESTION_PROMPT) 

doc_chain = load_qa_chain(llm_gen, chain_type="stuff", prompt=QA_PROMPT)

qa  = ConversationalRetrievalChain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )
# qa = ChatVectorDBChain(                                 
#     vectorstore=docsearch,
#     combine_docs_chain=doc_chain,
#     question_generator=question_generator,
#     return_source_documents = True, 
#     )

"""
Discovered ChatVectorDBChain (another langchain) while going through Langchain's Github. Helps us do exactly what we want. 
I had written a function for this process, but I feel doing it this way is faster and gives better results.

"""

chat_history = [] #Maintain a list for recording the chat history

def print_answer(question, chatHistoryFlag): 
    global chat_history
    if chatHistoryFlag == 1:
        chat_history = []
    result = qa(  #Run the chain
        {"question": question, "chat_history": chat_history},
        return_only_outputs=False,
        )
    
    chat_history.append((question, result['answer'])) #Update the chat history with the question asked and the answer
    #To create a chat history window
    if len(chat_history)>=5:
       chat_history=chat_history[1:] 

    #To get source from file metadata
    """ sources_set = set()
    for doc in result["source_documents"]:
        sources_set.add(doc.metadata['source']) 
    print("Sources:", sources_set) """

    return [result['answer']]
    # print(result)
# print(print_answer("who has apigee experience?"))
# print_answer("who has apigee experience?")