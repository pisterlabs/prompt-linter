from ._chroma import name2txt, get_chroma_ST

def get_chroma_self_query_retriever(_db_name):
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.llms import OpenAI
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo

    _metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Web link of the document",
            type="string",
        ),
        # AttributeInfo(
        #     name="title",
        #     description="Title of the document",
        #     type="string",
        # ),
        AttributeInfo(
            name="description",
            description="Description of the document",
            type="string",
        ),
        # AttributeInfo(
        #     name="language",
        #     description="Used language in the document",
        #     type="string",
        # ),
    ]
    llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _db = get_chroma_ST(_db_name)
    _document_contents = name2txt(_db_name)
    _retriever = SelfQueryRetriever.from_llm(llm, _db, _document_contents, _metadata_field_info, verbose=True)
    return _retriever

def qa_chroma_retriever_self_query(_query, _db_name):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    import os
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _retriever = get_chroma_self_query_retriever(_db_name)
    _qa_chain = RetrievalQA.from_chain_type(llm, retriever=_retriever)
    _ans = _qa_chain.run(_query)
    return _ans

def qa_chroma_self_query(_query, _db):
    _ans= ""
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _chroma_path = _pwd.parent.parent.parent
    _db_name = str(_chroma_path / "vdb" / _db)
    print(f"db_name: {_db_name}")
    _ans = qa_chroma_retriever_self_query(_query, _db_name)
    return _ans

def qa_chroma_self_query_azure(_query):
    _ans, _steps = "", ""
    _ans = qa_chroma_self_query(_query, "azure_virtual_machines_plus")
    return [_ans, _steps]

def qa_chroma_self_query_langchain(_query):
    _ans, _steps = "", ""
    _ans = qa_chroma_self_query(_query, "langchain_python_documents")
    return [_ans, _steps]


if __name__ == "__main__":

    _qa = [
        "how to save money on disk?",
        "how many disk types are billed per actually allocated disk size?",
        "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
        "can all disks be used to host operating systems for virtual machines?",
    ]
    for _q in _qa:
        _re= qa_chroma_self_query_azure(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")
    
    _qa = [
        "what's the difference between Agent and Chain in Langchain?"
    ]
    for _q in _qa:
        print(_q)
        _re= qa_chroma_self_query_langchain(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

