import os
import constants
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API
data_dir = "./data/grobid_files"
persist_dir = "./faiss_grobid"

def set_qa_prompt():
    """
    Define prompt template for QA
    """
    prompt = PromptTemplate(
        template=constants.QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt

def load_docs(data_dir):
    # Load the documents
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".xml"):
            loader = UnstructuredXMLLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def main():
    # If vector storage exists, load it. Otherwise, create one. 
    if not os.path.exists(persist_dir):
        chunked_documents = load_docs(data_dir)
        db = FAISS.from_documents(chunked_documents, embedding=OpenAIEmbeddings())
        os.makedirs(persist_dir)  # Create persist_dir before saving
        db.save_local(os.path.join(persist_dir, "faiss_index"))  # Save faiss_index inside persist_dir
    else:
        db = FAISS.load_local(os.path.join(persist_dir, "faiss_index"), embeddings=OpenAIEmbeddings())    
    
    # Load prompt template
    qa_prompt = set_qa_prompt()
    
    # Initialize retriever 
    retriever = db.as_retriever(search_kwargs={'k': 4})
    
    # Initialize QA chain
    qa = RetrievalQA.from_chain_type(
        llm=constants.llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt},
        
    ) 
    
    # Interactive questions and answers
    while True:
        query = input("\nAsk a question: ")
        if query == "exit" or query == "quit" or query == "q":
            break
        # Get the answer from the QA chain
        response = qa(query)
        answer, source_documents = response["result"], response["source_documents"]
        
        # Print the answer
        print("\n-> Answer:")
        print(answer)
        
        # Print the source documents if 'show_sources' is True
        for document in source_documents:
            print("\n> " + document.metadata["source"])
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    main()