import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import os, glob


# Sidebar contents
with st.sidebar:
    st.title("🤗💬 LLM Chat App")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [PaLM](https://makersuite.google.com/app/home) Embeddings & LLM model
 
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)")

load_dotenv()


def main():
    st.header("Chat with PDF 💬")

    files_path = "./SOURCE_DOCUMENTS/Orca_paper.pdf"
    loaders = [UnstructuredPDFLoader(files_path)]

    # if "index" not in st.session:
    index = VectorstoreIndexCreator(
        embedding=GooglePalmEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    ).from_loaders(loaders)

    llm = GooglePalm(temperature=0.1)  # OpenAI()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        # input_key="question",
        return_source_documents=True,
    )

    # st.session.index = index

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")
    # st.write(query)
    if query:
        response = chain(query)
        st.write(response["result"])
        with st.expander("Returned Chunks"):
            for doc in response["source_documents"]:
                st.write(f"{doc.metadata['source']} \n {doc.page_content}")


if __name__ == "__main__":
    main()
