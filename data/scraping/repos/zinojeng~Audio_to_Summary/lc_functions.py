from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI 
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from prompts import prompt_template, refine_prompt

# Create a function to split the text into chunks
def split_text(text, chunk_size, chunk_overlap):
    # Initialize text splitter
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=chunk_size, chunk_overlap= chunk_overlap)
    text_chunk = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in text_chunk]
    return docs

# Function initialize LLM
def initialize_llm(model, temperature, api_key):
    llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key = api_key)
    return llm

# Function to initialize OpenAI
def initialize_openai(model, temperature, api_key):
    llm2 = OpenAI(model=model, temperature=temperature, openai_api_key = api_key)
    return llm2

# Summarize the text
#def summarize_text(llm, docs):
    #initialize the chain
#    sum_chain = load_summarize_chain(
#        llm=llm, 
#        chain_type="refine", 
#        verbose=True,question_prompt=PROMPT)
#    summary = sum_chain.run(docs)
#    return summary

def summarize_text(llm, docs):
    #initialize the chain
    sum_chain = load_summarize_chain(
        llm=llm, 
        chain_type="refine", 
        verbose=True,
        question_prompt=prompt_template,
        refine_prompt=refine_prompt,
        )
    summary = sum_chain.run(docs)
    return summary