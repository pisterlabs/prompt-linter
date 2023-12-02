from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_txt(txt):
    text = txt.read().decode("utf-8")
    return text


def extract_text_from_brain():
    with open('brain/brain_journal.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def main():
    load_dotenv()
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Digital Brain Journal Search 🔍")
    st.write("Ask any questions about any of your journal entries with OpenAI's Embeddings and Langchain. The virtual brain keeps track of everything in a user's life. If you have another TXT or PDF file you'd like to search for answers, click on the dropdown and select eithter TXT or PDF option in file type. Along with the response, you will also get information about the amount of tokens that were used and the Total Cost of the query.")

    # Add API key input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
    else:
        file_type = st.selectbox("Choose the file type", options=["Brain", "PDF", "TXT"])

        file = None
        text = None

        if file_type == "PDF":
            file = st.file_uploader("Upload your PDF", type="pdf")
            if file is not None:
                text = extract_text_from_pdf(file)
        elif file_type == "TXT":
            file = st.file_uploader("Upload your TXT", type="txt")
            if file is not None:
                text = extract_text_from_txt(file)
        elif file_type == "Brain":
            text = extract_text_from_brain()

        if file is not None or file_type == "Brain":
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_area("Ask a question about your document:")

            if st.button("Submit"):
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)

                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)

                    st.markdown("### Response:")
                    st.write(response)
                    st.write(cb)
    st.markdown("---")
    st.markdown("")
    st.markdown("<p style='text-align: center'><a href='https://github.com/Kaludii'>Github</a> | <a href='https://huggingface.co/Kaludi'>HuggingFace</a></p>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
