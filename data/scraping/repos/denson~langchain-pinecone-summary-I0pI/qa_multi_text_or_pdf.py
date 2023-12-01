import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def main():

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")



    st.set_page_config(page_title="QA with a knowlege base", layout="wide")
    st.title("🤖! Question Answering with a knowlege base")

    st.markdown("""
    Enter your OpenAI API key. This costs $$. You will need to set up billing info at [OpenAI](https://platform.openai.com/account). \n
    Type your question at the bottom and click "Run" \n
    """)

    file_inputs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API key", type="password")
    pinecone_api_key = st.text_input("PINECONE_API_KEY", value="{}".format(PINECONE_API_KEY), type="password")        
    pinecone_env = st.text_input("PINECONE_ENVIRONMENT", value="{}".format(PINECONE_ENVIRONMENT))
    pinecone_index = st.text_input("PINECONE_INDEX", value="{}".format(PINECONE_INDEX))
    query = st.text_area("Enter your question")
    run_button = st.button("Run!")

    chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])
    k = st.slider("Number of relevant chunks", 1, 10, 2)

    if run_button and file_inputs and openai_key and query:
        os.environ["OPENAI_API_KEY"] = openai_key
        temp_files = []
        for file_input in file_inputs:
            temp_file = tempfile.NamedTemporaryFile(delete=False) 
            temp_file.write(file_input.getvalue())
            temp_file.close()
            temp_files.append(temp_file.name)
        
        result = qa(files=temp_files, query=query, chain_type=chain_type, k=k)
        st.write("🤖 Answer:", result["result"])
        st.write("Relevant source text:")
        st.write("--------------------------------------------------------------------\n".join(doc.page_content for doc in result["source_documents"]))

def qa(files, query, chain_type, k):
    documents = []
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    return result

if __name__ == "__main__":
    main()