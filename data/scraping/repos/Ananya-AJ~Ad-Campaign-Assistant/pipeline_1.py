import os
import openai
from langchain.llms import openai
from langchain.llms import openai as llms_openai
import pinecone
import uuid
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from doc_retireval import summarize_product_doc,create_ad_prompt
from prompt_db_module import generate_embedding,search_results,add_prompt
from llama_inference import *

API_KEY = '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'
PINECONE_API_KEY = 'eea250b5-8ade-4981-ab53-626dc466ac53'


def content_pipeline(prompt, strategy, domain, context):
    # Example flow (customize as needed):
    # 1. Generate embedding for a given prompt
    # embedding = generate_embedding(prompt)

    # 2. Search for results using the embedding
    matches = search_results(prompt)
    print("matches=",matches)

    # 3. Add a prompt to the index
    # add_prompt(prompt, prompt_type="type", domain="domain")

    # 4. Summarize a product document (provide the file path)
    # summary = summarize_product_doc('Data/pdfs/')

    # 5. Create an ad prompt
    product_query = prompt
    user_strategy = strategy
    user_structure = domain
    user_context = context
    product_found, final_prompt = create_ad_prompt(product_query, user_strategy, user_structure, user_context)
    return final_prompt


