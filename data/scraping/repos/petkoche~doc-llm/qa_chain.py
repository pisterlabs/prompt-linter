from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import os

def create_qa_chain(db, query):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={
                             "temperature": os.getenv("MODEL_TEMPERATURE"),
                             "max_length": os.getenv("MODEL_MAX_LENGTH")})

    chain = load_qa_chain(llm, chain_type="mydoc")
    docs = db.similarity_search(query)
    chain.run(input_documents=docs, question=query)
    # https://python.langchain.com/docs/modules/chains/additional/question_answering
