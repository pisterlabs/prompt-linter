def name2txt(_name):
    _1 = _name.split("_")
    _txt = " ".join(_1)
    return _txt

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def get_chroma_ST(_db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    _embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _db = Chroma(embedding_function=_embedding_function, persist_directory=_db_name)
    return _db

def get_chroma_OpenAI(_db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    _embedding_function = OpenAIEmbeddings()
    _db = Chroma(embedding_function=_embedding_function, persist_directory=_db_name)
    return _db

def similarity_search_chroma_ST(_query, _db_name):
    _db = get_chroma_ST(_db_name)
    _similar_docs = _db.similarity_search_with_score(_query)
    # _similar_docs = _db.similarity_search(_query)
    return _similar_docs

def get_chroma_multi_retrievers_chain():
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _chroma_path = _pwd.parent.parent.parent
    _db_azure = get_chroma_ST(str(_chroma_path / "vdb" / "azure_virtual_machines_plus"))
    _retriever_azure = _db_azure.as_retriever()
    _db_langchain = get_chroma_ST(str(_chroma_path / "vdb" / "langchain_python_documents"))
    _retriever_langchain = _db_langchain.as_retriever()
    _retrievers = [
        {
            "name": "azure virtual machines plus", 
            "description": "Good for answering general infomation about Azure virtual machines", 
            "retriever": _retriever_azure
        },
        {
            "name": "langchain python documents", 
            "description": "Good for answering general infomation about Langchain", 
            "retriever": _retriever_langchain
        }
    ]
    from langchain.chains.router import MultiRetrievalQAChain
    from langchain.llms import OpenAI
    _chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), _retrievers, verbose=True)
    return _chain

def qa_chroma_multi_retrievers(_query):
    _ans, _steps = "", ""
    _chain = get_chroma_multi_retrievers_chain()
    _ans = _chain.run(_query)
    print(_ans)
    return [_ans, _steps]

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()
    _qa = [
        # "how to save money on disk?",
        # "how many disk types are billed per actually allocated disk size?",
        # "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
        # "can all disks be used to host operating systems for virtual machines?",
        # "how disk types are billed?",
        # "how many disk types are billed on actually allocated disk size, and what are those disk types?",
        # "how many disk types are billed on buckets, and what are those disk types?",
        # "As you know, for some disk types, they are billed on actually allocated and used disk size; for other disk types, they are billed on its full bucket size, no matter how much exactly used. Please find out, in Azure, how many disk types are billed on actually allocated disk size, and what are those disk types? If you're lack of information, please say don't know.",
        # "As you know, for some disk types, they are billed on actually allocated and used disk size; for other disk types, they are billed on its full bucket size, no matter how much exactly used. Please find out, in Azure, how many disk types are billed on bucket size, and what are those disk types? If you're lack of information, please say don't know.",
        # "how many disk types are billed on buckets, and what are those disk types?",
        "how many disk types can not be used as OS disk, and what are they?",
    ]
    for _q in _qa:
        _re = qa_chroma_multi_retrievers(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

