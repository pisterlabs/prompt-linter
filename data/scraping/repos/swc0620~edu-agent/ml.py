from typing import Union
from time import time

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.schema.document import Document

## Template
from langchain.prompts import PromptTemplate

## Text Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Summary Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain


##### TEMPLATE ####
MAP_TEMPLATE = (
    """다음은 내용의 일부분이다.
    {docs}
    위 내용을 요약해줘:"""
)

REDUCE_TEMPLATE = (
    """다음은 내용을 요약한 것이다:
    {doc_summaries}
    위 내용들을 통해 전체적인 수업내용을 요약해줘.:"""
)

POSTPROCESS_TEMPLATE = (
    """다음 내용:
    {total_result}
    위 내용에서 중요한 부분 앞뒤에 **를 추가해서 다시 작성해줘:
    """
)

class SummaryModel():
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        ## 0. Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        llm = ChatOpenAI(temperature=0)

        # 1. Map chain 만들기
        map_prompt = PromptTemplate.from_template(MAP_TEMPLATE)
        self.map_chain = LLMChain(
            llm=llm, 
            prompt=map_prompt
        )

        # 2. Reduce chain 만들기
        reduce_prompt = PromptTemplate.from_template(REDUCE_TEMPLATE)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # 3. Combine Chain 만들기
        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, 
            document_variable_name="doc_summaries"
        )

        # 4. Reduce Chain 만들기
        # Combines and iteravely reduces the mapped documents
        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        # 5. Map Reduce Chain - Integration
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=self.reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=True,
        )
        
        # 6. Postprocess Chain
        postprocess_template = PromptTemplate.from_template(POSTPROCESS_TEMPLATE)
        self.postprocess_chain = LLMChain(
            llm=llm, 
            prompt=postprocess_template
        )
        

    def _split_text(
        self, 
        text: Union[str, list[str]]
    ) -> list[Document]:
        """
        string 혹은 list of string으로된 text를 여러개의 chunk로 나눕니다.
        
        이때 chunk는 Document에 대한 list 입니다.
        """
        if isinstance(text, list):
            chunks = self.text_splitter.create_documents(text)
        else:
            chunks = self.text_splitter.create_documents([text])
        
        return chunks
    
    def run(
        self, 
        text: Union[str, list[str]]
    ):
        """
        전체 요약 결과와 중간 요약 결과들을 리턴합니다.
        
        전체 요약 결과는 하나의 string 형태로 리턴되며 중간 요약 결과들은 list of string 형태로 리턴됩니다.
        
        Returns:
            total_result: str, intermediate_results: list[str]
        """
        
        chunks = self._split_text(text=text)
        
        start_time = time()
        total_result, intermediate_result_dict = self.map_reduce_chain.combine_docs(chunks)
        intermediate_results = intermediate_result_dict["intermediate_steps"]
        end_time = time()
        print(f"Map Reduce Time : {end_time-start_time}s")
        
        start_time = time()
        postprocessed_result = self.postprocess_chain.run(
            {"total_result": total_result}
        )
        end_time = time()
        print(f"Postprocess Time : {end_time-start_time}s")
        
        return postprocessed_result
