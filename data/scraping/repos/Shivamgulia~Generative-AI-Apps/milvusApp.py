from dotenv import load_dotenv
import os
import streamlit as st
from langchain.vectorstores.milvus import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pymilvus import connections, Collection

# initilizing the embedding model

from langchain.embeddings import HuggingFaceInstructEmbeddings


#  PROMPT Template

QuestionTemplate = """
Given the provided context, answer the following question. If the context does not mention any relevant information about the question, state "No relevant information found" and indicate the specific part of the context where the question should be addressed.

Context: {context}

Question: {query}

"""

prompt = PromptTemplate.from_template(QuestionTemplate)


# initilizing varibles

load_dotenv()


# connect to database

st.title("Document Question and Answer")

# ui components

uploadedDocument = st.file_uploader("Upload the file to question from")
button1 = st.button("Upload Document")


#  File Upload function
def uploadFile(file):
    if file is not None:
        documents = [file.read().decode()]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)
        return docs


def queryDocs():
    releventDocumnets = Milvus.similarity_search(
        db, query="what is generative artificial intelegence?"
    )
    st.write(releventDocumnets)


# file upload and generating embeddings

# global documents
# documents = None


def main():
    if button1:
        st.write("Document uploaded")
        global documents
        documents = uploadFile(uploadedDocument)
        st.write(documents[0])

        # Adding docks to Milvus vectorstore

        print(documents)
        print("generateing embeddings ....")
        st.write("generateing embeddings ....")
        instEmbedder = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
        )
        st.write(documents[0])
        db = Milvus(
            embedding_function=instEmbedder,
            connection_args={"host": "127.0.0.1", "port": "19530"},
            collection_name="Application",
        )
        db.add_documents(documents=documents)
        print("embeddings stored")
        st.write("embeddings stored")

    # taking query and generating response

    question = st.text_input("Question")

    if st.button("Answer"):
        if len(question) <= 1:
            st.write("write a question first")

        if len(question) > 1:
            st.write(question)
            instEmbedder = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
            )
            db = Milvus(
                embedding_function=instEmbedder,
                connection_args={"host": "127.0.0.1", "port": "19530"},
                collection_name="Application",
            )
            # documents = db.similarity_search(query=question)
            # documentData = ""
            # for doc in documents:
            #     documentData += doc.page_content
            #     documentData += "\n"
            # questionPrompt = prompt.format(query=question, context=documentData)
            # st.write(questionPrompt)

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0.7),
                chain_type="stuff",
                retriever=db.as_retriever(),
                verbose=True,
            )

            response = qa.run(question)
            st.write(response)


if __name__ == "__main__":
    main()
