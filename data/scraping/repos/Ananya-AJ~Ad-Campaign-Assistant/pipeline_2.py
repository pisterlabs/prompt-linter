import os
import openai
from langchain.llms import openai
from langchain.llms import openai as llms_openai
import pinecone
import uuid
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from doc_retireval import summarize_product_doc,create_ad_prompt
from prompt_db_module import generate_embedding,search_results,add_prompt
from llama_inference import invoke_inference

API_KEY = 'apikey'


def ad_pipeline(prompt):
    ad,banners = invoke_inference(prompt)
    return ad, banners

