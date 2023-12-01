from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Megazone Cloud ChatBot")
    st.markdown("<h1 style='text-align: center; color: lightgreen;'>Megazone Cloud ChatBot 💬</h1>",
                unsafe_allow_html=True)

    # Set Azure environment variables
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    os.environ["OPENAI_API_BASE"] = "https://mtcaichat01.openai.azure.com"
    os.environ["OPENAI_API_KEY"] = "824fe43e851f4862af326fa83c3d3cfe"

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        with st.spinner('Reading the PDF...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            with st.spinner('Creating knowledge base...'):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_input("Ask a question 🤖:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                # replace deployment_name and model_name with your own
                llm = AzureOpenAI(
                    deployment_name="embeddingada002",
                    model_name="text-embedding-ada-002",
                )
                chain = load_qa_chain(llm, chain_type="stuff")

                with st.spinner('Generating answer...'):
                    with get_openai_callback() as cb:
                        response = chain.run(
                            input_documents=docs, question=user_question)
                        print(cb)

                # Display the result in a more noticeable way
                st.markdown(
                    f'### Answer: \n {response}', unsafe_allow_html=True)


if __name__ == '__main__':
    main()


# from dotenv import load_dotenv
# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import AzureOpenAI
# from langchain.callbacks import get_openai_callback


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Megazone Cloud ChatBot")
#     st.header("Megazone Cloud ChatBot 💬")

#     # Set Azure environment variables (replace "..." with actual values)
#     os.environ["OPENAI_API_TYPE"] = "azure"
#     os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
#     os.environ["OPENAI_API_BASE"] = "https://mtcaichat01.openai.azure.com"
#     os.environ["OPENAI_API_KEY"] = "824fe43e851f4862af326fa83c3d3cfe"

#     # upload file
#     pdf = st.file_uploader("Upload your PDF", type="pdf")

#     # extract the text
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         # split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # create embeddings
#         embeddings = OpenAIEmbeddings()
#         knowledge_base = FAISS.from_texts(chunks, embeddings)

#         # show user input
#         user_question = st.text_input("Ask a question 🤖:")
#         if user_question:
#             docs = knowledge_base.similarity_search(user_question)

#             # replace deployment_name and model_name with your own
#             llm = AzureOpenAI(
#                 deployment_name="",
#                 model_name="text-davinci-002",
#             )
#             chain = load_qa_chain(llm, chain_type="stuff")
#             with get_openai_callback() as cb:
#                 response = chain.run(input_documents=docs,
#                                      question=user_question)
#                 print(cb)

#             st.write(response)


# if __name__ == '__main__':
#     main()
