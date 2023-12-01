import pandas as pd

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import logging
from rich import print
from rich.logging import RichHandler
from rich.pretty import pretty_repr

import os
from dotenv import load_dotenv

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")

load_dotenv() # load .env file
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY") # "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

def faiss_answer(documents, input, prompt):
    log.debug("%s", "Starting...",)

    # create a dataframe from documents
    df = pd.DataFrame(documents)
    log.debug("Documents:\n%s", df)

    # vectorizing documents using 'question' column as page content
    loader = DataFrameLoader(df, page_content_column='question')
    documents = loader.load()

    # create text splitter for fit to chunks size (not very useful for this example)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # initialize embeddings, we will use OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

    # create vector store
    db = FAISS.from_documents(texts, embeddings)
    db.as_retriever()

    # # if needed, save index locally 
    # db.save_local('faiss_index')

    # ranking documents by similarity to input
    ranked_docs = db.similarity_search_with_score(input)
    log.debug("ranked_docs:\n%s", pretty_repr(ranked_docs))
    
    doc = ranked_docs[0][0].dict()['metadata']
    doc['input'] = input # here we add original input to the doc
    log.debug("doc:\n%s", pretty_repr(doc))

    # chain with custom prompt
    chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=OPEN_AI_KEY, max_tokens=500),
        prompt=prompt)
    log.debug("chain:\n%s", pretty_repr(chain))

    answer = chain.run(doc)
    log.debug("answer:\n%s", pretty_repr(answer))

    return(answer)

    #### HERE IS ONE MORE POSSIBLE IMPLEMENTATION USING RetrievalQA:
    #
    # # create a chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=OpenAI(temperature=0, openai_api_key=OPEN_AI_KEY),
    #     chain_type='stuff',
    #     retriever=db.as_retriever()
    # )

    # log.info("qa_chain:\n%s", pretty_repr(qa_chain))
    # # print(qa_chain)
    #
    #####

def run():
    # inputs for the test search - try different
    
    input = "I forgot my password."
    # input = "I'm concerned about the safety of my account."
    # input = "What is bla-bla-bla."
    # input = "Call a human, please"

    # create a list of example documents
    documents = [
        {"id": 1, "question": "How to recover the password?", "answer": "To reset your password, please click on the 'Forgot Password?' link on the login page. Enter your email address, and we will send you instructions on how to reset your password.", "url": "https://example.com/confluence/recover-password"},
        {"id": 2, "question": "How can I contact the support service?", "answer": "You can contact the support service by emailing us at support@example.com or by calling us at +1 (123) 456-7890.", "url": "https://example.com/confluence/contact-support"},
        {"id": 3, "question": "How to set up two-factor authentication?", "answer": "To set up two-factor authentication, go to the 'Security Settings' section of your account and follow the instructions.", "url": "https://example.com/confluence/2fa-setup"},
    ]

    # create a prompt template
    prompt_template = """
    Compose the answer for the input, following these rules:
    - Use the following pieces of context to answer the question at the end.
    - You can slightly change the answer adding the goal from the input, but do not change the general sense of the answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    - At the end, be sure to add the link with url to the full document. You can miss the link you don't know the answer.
    input: {input}
    context: {answer}
    url: {url}
    Helpful Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=['input', 'answer', 'url']
    )

    try:
        print(faiss_answer(documents=documents, input=input, prompt=prompt))
    except Exception:
        log.exception("Unexpected error")
        raise


if __name__ == "__main__":
    run()