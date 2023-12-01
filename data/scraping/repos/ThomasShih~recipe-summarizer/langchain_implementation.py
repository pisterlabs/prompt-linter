__all__ = ["summarize_recipe", "extract_from_url"]

import os
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.chains import SimpleSequentialChain
from langchain.schema import SystemMessage

llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"))

def extract_from_url(url: str) -> str:
    loader = AsyncHtmlLoader(url)
    html2text = Html2TextTransformer()

    docs = loader.load()
    docs_transformed = html2text.transform_documents(docs)
    # TODO: We don't want to do this, lets figure out a different way
    content = docs_transformed[0].page_content[:4000]

    return content

llm_chain = LLMChain(
    prompt=PromptTemplate(
        template="Using {page_content}, summarize the steps needed to make this recipe.", 
        input_variables=["page_content"],
    ), 
    llm=llm,
)

llm_chain_2 = LLMChain(
    prompt=PromptTemplate(
        template="further summarize {recipe_details} include only details that a professional chef would need to know.", 
        input_variables=["recipe_details"],
    ), 
    llm=llm,
)

llm_chain_3 = LLMChain(
    prompt=PromptTemplate(
        template="If its not in bullet form, convert {recipe_details} into bullet form.", 
        input_variables=["recipe_details"],
    ), 
    llm=llm,
)

sequential_chain = SimpleSequentialChain(
    chains=[llm_chain, llm_chain_2, llm_chain_3],
    verbose=True,
)

def summarize_recipe(page_content: str):
    return sequential_chain.run(page_content)