import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.llms import Replicate
from dotenv import load_dotenv

load_dotenv()


def load_data():
    loader = CSVLoader(file_path="customer_response_data.csv")
    return loader.load()


def create_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(documents, embeddings)


def retrieve_info(query, db):
    similar_response = db.similarity_search(query, k=3)
    page_contents = [doc.page_content for doc in similar_response]
    return page_contents


def create_prompt_template():
    template = """
    You are an exceptional business development representative.
    When I provide you with a prospect's message, your task is
    to craft the most effective response tailored in a letter format
    to the prospect's feedback while adhering to the following guidelines:

    1. Response should be very similar or even identical to the past responses,
    in terms of length, tone of voice, logical arguments, and other details

    2. If the responses are irrelevant, then try to mimic the style of the 
    prospect response.

    below is a message I receive from the prospect:
    {message}

    Here is a list of responses of how we normally respond to prospects in a similar scenario
    {prospect_responses}

    please write the best response that I should send to this prospect:
    """
    return PromptTemplate(
        input_variables=["message", "prospect_responses"], template=template
    )


def create_llm_chain():
    llm = Replicate(
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1},
    )
    # llm = CTransformers(
    # model = "llama-2-7b-chat.ggmlv3.q4_0.bin",
    # model_type="llama",
    # max_new_tokens = 512,
    # temperature = 0.5)

    prompt = create_prompt_template()
    return LLMChain(llm=llm, prompt=prompt)


def generate_response(message, db, chain):
    prospect_responses = retrieve_info(message, db)
    return chain.run(message=message, prospect_responses=prospect_responses)


def main():
    st.header("Customer Response Generator :books:")
    message = st.text_area("Customer Message")
    response = st.empty()

    if st.button("Send"):
        if message:
            response.write("Generating message....")
            documents = load_data()
            db = create_embeddings(documents)
            chain = create_llm_chain()
            result = generate_response(message, db, chain)
            response.info(result)


if __name__ == "__main__":
    main()
