from repolya._log import logger_paper

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from langchain.document_transformers import (
    LongContextReorder,
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)
from langchain.chains import (
    RetrievalQA,
    RetrievalQAWithSourcesChain,
    StuffDocumentsChain,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from typing import List
from pydantic import BaseModel, Field
import os


# _chain_type = "map_reduce" # stuff, map_reduce, refine, map_rerank

def pretty_print_docs(docs):
    # _pretty = f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])
    _doc = []
    _doc_meta = []
    for i, doc in enumerate(docs):
        _doc.append(f"Document {i+1}:\n\n" + str(doc.metadata) + "\n\n" + doc.page_content)
        _doc_meta.append(f"Document {i+1}:\n\n" + str(doc.metadata)+ "\n" + str(len(doc.page_content)))
    _pretty = f"\n{'-' * 60}\n".join(_doc)
    # print(_pretty)
    _meta = f"\n{'-' * 60}\n".join(_doc_meta)
    # print(_meta)
    return _pretty


##### OpenAI retriever
def get_faiss_OpenAI(_db_name):
    _embeddings = OpenAIEmbeddings()
    _db = FAISS.load_local(_db_name, _embeddings)
    return _db

def get_faiss_OpenAI_multi_query_retriever(_db_name):
    class LineList(BaseModel):
        # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)
        def parse(self, text: str) -> LineList:
            lines = text.strip().split("\n")
            return LineList(lines=lines)
    output_parser = LineListOutputParser()
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions seperated by newlines.
Original question: {question}""",
    )
    llm = ChatOpenAI(
        model_name=os.getenv('OPENAI_API_MODEL'),
        temperature=0
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=QUERY_PROMPT,
        output_parser=output_parser
    )
    _vdb = get_faiss_OpenAI(_db_name)
    _base_retriever = _vdb.as_retriever()
    ##### Remove redundant results from the merged retrievers
    _filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings())
    ##### Re-order results to avoid performance degradation
    _reordering = LongContextReorder()
    ##### ContextualCompressionRetriever
    _pipeline = DocumentCompressorPipeline(transformers=[_filter, _reordering])
    _compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=_pipeline,
        base_retriever=_base_retriever
    )
    ##### MultiQueryRetriever
    _multi_retriever = MultiQueryRetriever(
        retriever=_compression_retriever_reordered,
        llm_chain=llm_chain,
        parser_key="lines"
    )
    return _multi_retriever

def qa_faiss_OpenAI_multi_query(_query, _db_name, _chain_type):
    _ans, _steps = "", ""
    # llm = ChatOpenAI(model_name=os.getenv('OPENAI_API_MODEL'), temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    with get_openai_callback() as cb:
        _multi_retriever = get_faiss_OpenAI_multi_query_retriever(_db_name)
        _run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        _generated_queries = _multi_retriever.generate_queries(_query, _run_manager)
        logger_paper.info(f"Q: {_query}")
        for i in _generated_queries:
            logger_paper.info(i)
        ##### _docs
        _docs = _multi_retriever.get_relevant_documents(_query)
        #####
        _qa = load_qa_chain(
            llm,
            chain_type=_chain_type
        )
        _ans = _qa(
            {"input_documents": _docs, "question": _query},
            return_only_outputs=True
        )
        #####
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        # print(_token_cost)
        _steps = f"{_token_cost}\n\n"+ "\n".join(_generated_queries)
        _steps += f"\n\n{'=' * 40}docs\n" + pretty_print_docs(_docs)
        logger_paper.info(f"A: {_ans['output_text']}")
        logger_paper.info(f"[{_chain_type}] {_token_cost}")
        logger_paper.debug(f"[{_chain_type}] {_steps}")
    return [_ans['output_text'], _steps]


##### ST retriever
def get_faiss_ST(_db_name):
    ### all-MiniLM-L12-v2
    _db_name_all = os.path.join(_db_name, 'all-MiniLM-L12-v2')
    _embedding_all = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    _db_all = FAISS.load_local(_db_name_all, _embedding_all)
    ### multi-qa-mpnet-base-dot-v1
    _db_name_multiqa = os.path.join(_db_name, 'multi-qa-mpnet-base-dot-v1')
    _embedding_multiqa = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    _db_multiqa = FAISS.load_local(_db_name_multiqa, _embedding_multiqa)
    return _db_all, _db_multiqa

def get_faiss_ST_multi_query_retriever(_db_name):
    class LineList(BaseModel):
        # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)
        def parse(self, text: str) -> LineList:
            lines = text.strip().split("\n")
            return LineList(lines=lines)
    output_parser = LineListOutputParser()
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions seperated by newlines.
Original question: {question}""",
    )
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_API_MODEL'), temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
    ##### 
    _db_all, _db_multiqa = get_faiss_ST(_db_name)
    _retriever_all = _db_all.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "include_metadata": True}
    )
    _retriever_multiqa = _db_multiqa.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "include_metadata": True}
    )
    _lotr = MergerRetriever(retrievers=[_retriever_all, _retriever_multiqa])
    ##### Remove redundant results from the merged retrievers
    _filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings())
    ##### Re-order results to avoid performance degradation
    _reordering = LongContextReorder()
    ##### ContextualCompressionRetriever
    _pipeline = DocumentCompressorPipeline(transformers=[_filter, _reordering])
    _compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=_pipeline,
        base_retriever=_lotr
    )
    ##### MultiQueryRetriever
    _multi_retriever = MultiQueryRetriever(
        retriever=_compression_retriever_reordered,
        llm_chain=llm_chain,
        parser_key="lines"
    )
    return _multi_retriever

def qa_faiss_ST_multi_query(_query, _db_name, _chain_type):
    _ans, _steps = "", ""
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_API_MODEL'), temperature=0)
    with get_openai_callback() as cb:
        _multi_retriever = get_faiss_ST_multi_query_retriever(_db_name)
        _run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        _generated_queries = _multi_retriever.generate_queries(_query, _run_manager)
        logger_paper.info(f"Q: {_query}")
        for i in _generated_queries:
            logger_paper.info(i)
        ##### _docs
        _docs = _multi_retriever.get_relevant_documents(_query)
        #####
        _qa = load_qa_chain(
            llm,
            chain_type=_chain_type
        )
        try:
            _ans = _qa(
                {"input_documents": _docs, "question": _query},
                return_only_outputs=True
            )
        except Exception as e:
            logger_paper.debug(f"{e}")
            _chain_type = 'refine'
            _qa = load_qa_chain(
                llm,
                chain_type=_chain_type
            )
            logger_paper.info(f"start [refine]")
            _ans = _qa(
                {"input_documents": _docs, "question": _query},
                return_only_outputs=True
            )
        #####
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        # print(_token_cost)
        _steps = f"{_token_cost}\n\n"+ "\n".join(_generated_queries)
        _steps += f"\n\n{'=' * 40} docs\n" + pretty_print_docs(_docs)
        logger_paper.info(f"A: {_ans['output_text']}")
        logger_paper.info(f"[{_chain_type}(lotr)] {_token_cost}")
        logger_paper.debug(f"[{_chain_type}(lotr)] {_steps}")
    return [_ans['output_text'], _steps]

