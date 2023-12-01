import os
from dotenv import find_dotenv,load_dotenv

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

from src.prompts import map_prompt_template, combine_prompt_template

# load the api key from the .env file
load_dotenv(find_dotenv())
openai_api_key=os.getenv("OPENAI_API_KEY")

# extract text from pdf
def extract_text_from_pdf(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)
    docs = loader.load_and_split()
    return docs

# summarize pdf
def summarize_pdf(path_to_pdf):

    # extract text from pdf
    docs = extract_text_from_pdf(path_to_pdf)

    # load the language model and the chain
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     verbose=True
                                    )

    # run the chain
    summary = summary_chain.run(docs)

    return summary
