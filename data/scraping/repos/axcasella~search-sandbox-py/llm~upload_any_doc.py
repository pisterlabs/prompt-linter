import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate, ChatPromptTemplate)

import pickle
import os

# Main page contents
def main():
    load_dotenv()

    st.header("Simple search of 10-Ks and 10-Qs")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    vector_store, store_file_name = create_vector_store_from_pdf(pdf)
    if vector_store and store_file_name is not None:
        num_vectors = vector_store.index.ntotal
        st.write(f"New file, created vector store with {num_vectors} vectors")

        with open(f"{store_file_name}.pk1", "wb") as f:
            pickle.dump(vector_store, f)
            
        # Take in search query
        query = st.text_input("Search for:")
        if query:
            # Get top results
            k = 3
            response, docs = get_response_from_query(vector_store, query, k=k)
            st.write("Answer:")
            st.write(response)

            st.write(f"Top {k} results:")
            cleaned_docs = [doc.page_content.replace('\n', ' ') for doc in docs]
            st.write(cleaned_docs)

def create_vector_store_from_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Get rid of .pdf from name
        store_file_name = pdf.name[:-4]

        if os.path.exists(f"{store_file_name}.pk1"):
            # read the file
            with open(f"{store_file_name}.pk1", "rb") as f:
                vectorStore = pickle.load(f)

            st.write('File already exists, loaded embeddings from disk')
        else:
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Use langchain to split into chunks
            splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text=text)
            st.write("File broken into chunks: ", len(chunks))
            st.write("Chunk:")
            st.write(chunks)

            # Embed chunks
            embeddings = OpenAIEmbeddings(model="chatgpt-3.5-turbo")

            # Create vector store using Meta's FAISS store
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        return vectorStore, store_file_name
    else: 
        return None, None

def get_response_from_query(db, query, k=3):
    docs = db.similarity_search(query=query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    # Ask LLM to give final result
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=3000)

    system_template = """
        Only use factual information from the document {docs}.

        If you don't have enough information, just say "I don't know".
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
        Anser the question: {question}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(docs=docs_page_content, question=query)

    return response, docs

if __name__ == "__main__":
    main()