from langchain import PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain, BaseCombineDocumentsChain
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
import asyncio
from langchain.chains.question_answering.map_reduce_prompt import COMBINE_PROMPT
from llm.llm import llm
import os
from langchain.chains import LLMChain
from prompt.Extraction import EXTRACTION_PROMPT, REFINE_PROMPT


class ExtractionServices:
    conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                            "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
    container_name = os.getenv("CONTAINER_NAME", "gptfiles")

    @classmethod
    def extract_entity(cls, content):
        prompt = PromptTemplate.from_template(EXTRACTION_PROMPT)
        chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        content = chain.run(content)
        return content

    @classmethod
    async def run_chain(cls, content):
        prompt = PromptTemplate.from_template(EXTRACTION_PROMPT)
        chain = LLMChain(propmt=prompt, llm=llm, verbose=True)
        content = chain.run(content)
        return content
