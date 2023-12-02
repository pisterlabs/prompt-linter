from pathlib import Path

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from ..basetoolboxnode import BaseToolboxNode
from wavesynlib.languagecenter.wavesynscript import (
    Scripting, ModelNode, WaveSynScriptAPI, NodeDict, code_printer)



class LangChainNode(ModelNode):
    def summarize(self, text):
        llm = OpenAI(temperature=0)
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(docs)


class ToolboxNode(BaseToolboxNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langchain = LangChainNode()

