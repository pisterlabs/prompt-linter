import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma


# Define bot avatar display function
def display_avatar():
    st.image("avatar/bot_avatar.jpeg", width=100)


def main():
    # Streamlit app
    st.title('AI-Powered Virtual Assistant')

    # Load OpenAI API key from .env file
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    # Get source document input
    user_doc = st.file_uploader("Upload Your PDF Document", type="pdf")

    # Get user question input
    user_question = st.text_input("Ask a question about your document")

    # Check if the 'Ask' button is clicked
    if st.button("Enter"):
        # Validate input
        if not user_doc or not user_question:
            st.write(f"Please upload a PDF document and enter a question.")
        else:
            try:
                # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(user_doc.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)

                # Define the template for the language model prompt
                template = """You are a language model AI developed for summarizing files. \
                            You are a friendly virtual assistant designed to provide information based on documents. \
                            Your objective is to provide accurate responses to users \
                            questions, strictly based on the documents they upload. \

                            In your responses, ensure a tone of friendliness but professionalism. \

                            Here are some specific interaction scenarios to guide your responses:
                            - If the user asks what you can do, respond with "I'm a Virtual Assistant here to provide \
                            you with information about your file. How can I assist you?"
                            - If the user starts with a greeting, respond with 'Hello! How are you doing today? \
                            How can I assist you?' or something related to that
                            - If a user shares their name, use it in your responses when appropriate, to cultivate a \
                            more personal and comforting conversation.
                            - If a user poses a question about their document, answer based on their document only.
                            - If a user asks a question that is unrelated to their document, respond with \
                            'Sorry, I'm built to only answer questions related to your document. \
                            Could you please ask a question related to your document?'

                            {context}
                            Question: {question}
                            Answer:"""

                # Create a prompt template
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])

                # Create embeddings for the pages and insert into Chroma database
                embeddings = OpenAIEmbeddings()
                vectordb = Chroma.from_documents(pages, embeddings)
                retriever = vectordb.as_retriever()
                chain_type_kwargs = {"prompt": prompt}

                # Initialize the OpenAI module, load and run the retrieval QA chain
                llm = OpenAI(temperature=0)
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    verbose=True
                )
                search = vectordb.similarity_search(user_question)
                answer = chain.run(input_documents=search, query=user_question)

                # Display bot avatar and chatbot response
                display_avatar()
                st.markdown(f"**Lazer:** {answer}")

            except Exception as e:
                st.write(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
