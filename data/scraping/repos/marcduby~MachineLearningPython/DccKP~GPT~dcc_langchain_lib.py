
# imports
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import CTransformers

import time 
import os

# constants
BEGIN_INST, END_INST = "[INST]", "[/INST]"
BEGIN_SYS, END_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_SYSTEM_DEFAULT = """
"""

# prompt
PROMPT_GENE_TEXT = """
Below are the abstracts from different research papers on gene {gene}. 
Please read through the abstracts and as a geneticist write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {gene}
{text}
"""
PROMPT_SIMPLE_GENE_TEXT = """
Please read through the following text and as a geneticist write a 100 word summary that synthesizes the key findings on the biology of gene {gene}
{text}
"""
PROMPT_SIMPLE_BULLETPOINT_GENE_TEXT = """
Please read through the following text and as a geneticist write a 100 word bullet point summary that synthesizes the key findings on the biology of gene {gene}
{text}
"""

# instructions
INSTRUCTION_AS_GENETICIST = """
Below are the abstracts from different research papers on gene {}. 
Please read through the abstracts and write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {}
{}
"""

# roles
ROLE_BIOLOGIST_IN_GENETICS = "You are an biologist that excels at genetics research."


# prompts
TEMPLATE_PROMPT_EMPTY = "{text_prompt}"

# files
DIR_DATA = "/home/javaprog/Data/"
FILE_LLAMA2_7B_CPU = DIR_DATA + "ML/Llama2Test/Model/llama-2-7b-chat.ggmlv3.q8_0.bin"
FILE_LLAMA2_13B_CPU = DIR_DATA + "ML/Llama2Test/Model/llama-2-13b-chat.ggmlv3.q8_0.bin"


# langchain methods
def get_template_prompt(template_prompt, log=False):
    '''
    will create a lanchain prompt
    '''
    prompt = template_prompt
    prompt_template = PromptTemplate.from_template(prompt)
    return prompt_template

def get_summarize_chain(llm, chain_type='map_reduce', verbose=False, log=False):
    '''
    loads a summarization chain
    '''
    chain = load_summarize_chain(llm=llm, chain_type=chain_type, verbose=verbose)
    return chain

def create_docs_list_from_text_list(list_text, log=False):
    '''
    create a docs list for chain use
    '''
    docs = [Document(page_content=t) for t in list_text]
    return docs

def split_text_to_chunks(text, chunk_size=1000, chunk_overlap=20, log=False):
    '''
    will use the text splitter to split text into list of text
    '''
    text_splitter = CharacterTextSplitter(separator='\n',
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text)

    if log:
        print("got chunk list of size: {}".format(len(text_chunks)))

    return text_chunks



# instruction methods
def get_prompt_with_chat_tags(instruction, system_prompt, log=False):
    '''
    will create a prompt with instruction and system prompt
    '''
    SYSTEM_PROMPT = BEGIN_SYS + system_prompt + END_SYS
    prompt_template =  BEGIN_INST + SYSTEM_PROMPT + instruction + END_INST
    return prompt_template

def get_prompt_no_tags(instruction, system_prompt, log=False):
    '''
    will create a prompt with instruction and system prompt
    '''
    SYSTEM_PROMPT = system_prompt + "\n"
    prompt_template =  SYSTEM_PROMPT + instruction
    return prompt_template

def get_biology_summary_instruction(gene, text, log=False):
    '''
    will create a genetics summary instruction
    '''
    instruction = INSTRUCTION_AS_GENETICIST.format(gene, gene, text)

    return instruction


# model methods
def load_local_llama_model(file_model, temperature=0.1, max_new_tokens=512, log=False):
    '''
    load the model from the file
    '''
    if log:
        print("loading model: {}".format(file_model))

    llm = CTransformers(
        model=file_model,
        model_type = "llama",
        max_new_tokens = max_new_tokens,
        temperature = temperature
    )

    if log:
        print("loaded model from: {}".format(file_model))

    return llm



