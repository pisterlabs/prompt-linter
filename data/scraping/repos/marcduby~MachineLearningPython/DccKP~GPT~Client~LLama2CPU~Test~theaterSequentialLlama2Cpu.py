

# imports
from langchain import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.chains import SequentialChain
import time 
import os

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# local imports
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib
import dcc_langchain_lib

# main
if __name__ == "__main__":
    # title = input('What is your title? ')
    title = "where's my car?"
    print ('I have your title as ' + title)

    # era = input('Lasty, from what era should the text be? ')
    era = "Roaring Twenties"
    print ('I have your era as ' + era)

    # This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
    # llm = OpenAI(temperature=.7)
    llm = dcc_langchain_lib.load_local_llama_model(file_model=dcc_langchain_lib.FILE_LLAMA2_13B_CPU)
    template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""
    prompt_template = PromptTemplate(input_variables=["title", 'era'], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

    # This is an LLMChain to write a review of a play given a synopsis.
    # llm = OpenAI(temperature=.7)
    # llm = dcc_langchain_lib.load_local_llama_model(file_model=dcc_langchain_lib.FILE_LLAMA2_13B_CPU)
    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    # This is the overall chain where we run these two chains in sequence.
    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True)


    review = overall_chain({"title":title, "era": era})
    print (review) 


