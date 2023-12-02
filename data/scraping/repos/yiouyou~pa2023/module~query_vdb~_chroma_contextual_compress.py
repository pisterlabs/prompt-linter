from ._chroma import get_chroma_ST, pretty_print_docs

def get_chroma_contextual_compress_retriever(_db_name):
    import os
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.document_transformers import EmbeddingsRedundantFilter
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.retrievers import ContextualCompressionRetriever

    _embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0, separator=". ")
    _redundant_filter = EmbeddingsRedundantFilter(embeddings=_embedding_function)
    _relevant_filter = EmbeddingsFilter(embeddings=_embedding_function, similarity_threshold=0.76)
    _pipeline_compressor = DocumentCompressorPipeline(
        transformers=[_splitter, _redundant_filter, _relevant_filter]
    )
    _db = get_chroma_ST(_db_name)
    _base_retriever = _db.as_retriever()
    _compression_retriever = ContextualCompressionRetriever(base_compressor=_pipeline_compressor, base_retriever=_base_retriever)

    # from langchain.llms import OpenAI
    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain.retrievers.document_compressors import LLMChainExtractor

    # llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    # _compressor = LLMChainExtractor.from_llm(llm)
    # _compression_retriever = ContextualCompressionRetriever(base_compressor=_compressor, base_retriever=_base_retriever)

    # from langchain.retrievers.document_compressors import LLMChainFilter
    # _filter = LLMChainFilter.from_llm(llm)
    # _compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=_base_retriever)

    return _compression_retriever

def qa_chroma_retriever_contextual_compress(_query, _db_name):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    import os
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _retriever = get_chroma_contextual_compress_retriever(_db_name)
    # _docs = _retriever.get_relevant_documents(_query)
    # pretty_print_docs(_docs)
    _qa_chain = RetrievalQA.from_chain_type(llm, retriever=_retriever)
    _ans = _qa_chain.run(_query)
    return _ans

def qa_chroma_contextual_compress(_query, _db):
    _ans= ""
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _chroma_path = _pwd.parent.parent.parent
    _db_name = str(_chroma_path / "vdb" / _db)
    print(f"db_name: {_db_name}")
    _ans = qa_chroma_retriever_contextual_compress(_query, _db_name)
    return _ans

def qa_chroma_contextual_compress_azure(_query):
    _ans, _steps = "", ""
    _ans = qa_chroma_contextual_compress(_query, "introduction_to_azure_managed_disks")
    return [_ans, _steps]

def qa_chroma_contextual_compress_langchain(_query):
    _ans, _steps = "", ""
    _ans = qa_chroma_contextual_compress(_query, "langchain_python_documents")
    return [_ans, _steps]


if __name__ == "__main__":

    from _faiss import get_faiss_ST, pretty_print_docs

    _qa = [
        "how to save money on disk?",
        "how many disk types are billed per actually allocated disk size?",
        "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
        "can all disks be used to host operating systems for virtual machines?",
    ]
    for _q in _qa:
        _re= qa_chroma_contextual_compress_langchain(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

    _qa = [
        "what's the difference between Agent and Chain in Langchain?"
    ]
    for _q in _qa:
        print(_q)
        _re= qa_chroma_contextual_compress_langchain(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

