from ._faiss import get_faiss_ST, pretty_print_docs

def get_faiss_vdb_retriever(_db_name):
    _db = get_faiss_ST(_db_name)
    # _vdb_retriever = _db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
    _vdb_retriever = _db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    return _vdb_retriever

def qa_faiss_retriever_vdb(_query, _db_name):
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    import os
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _retriever = get_faiss_vdb_retriever(_db_name)
    # _docs = _retriever.get_relevant_documents(_query)
    # pretty_print_docs(_docs)
    ##### RetrievalQA
    # from langchain.chains import RetrievalQA
    # _qa_chain = RetrievalQA.from_chain_type(llm, retriever=_retriever)
    # _ans = _qa_chain.run(_query)
    ##### RetrievalQAWithSourcesChain
    from langchain.chains import RetrievalQAWithSourcesChain
    _qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=_retriever, reduce_k_below_max_tokens=True)
    _ans = _qa_chain({"question": _query}, return_only_outputs=True)
    _ans = f"{_ans['answer'].strip()} ({_ans['sources']})"
    return _ans

def qa_faiss_vdb(_query, _db):
    _ans= ""
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _faiss_path = _pwd.parent.parent.parent
    _db_name = str(_faiss_path / "vdb" / _db)
    print(f"db_name: {_db_name}")
    _ans = qa_faiss_retriever_vdb(_query, _db_name)
    return _ans

def qa_faiss_vdb_azure(_query):
    _ans, _steps = "", ""
    _ans = qa_faiss_vdb(_query, "azure_vm")
    return [_ans, _steps]

# def qa_faiss_vdb_langchain(_query):
#     _ans, _steps = "", ""
#     _ans = qa_faiss_vdb(_query, "langchain_python_documents")
#     return [_ans, _steps]


if __name__ == "__main__":

    from _faiss import get_faiss_ST, pretty_print_docs

    _qa = [
        # "how to save money on disk?",
        # "how disk types are billed?",
        # "how many disk types are billed on actually allocated disk size, and what are those disk types?",
        # "how many disk types are billed on buckets, and what are those disk types?",
        # "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
        # "what disk types are billed per actually allocated disk size?",
        # "what disk types are billed in buckets?",
        # "As you know, for some disk types, they are billed on actually allocated and used disk size; for other disk types, they are billed on its full bucket size, no matter how much exactly used. Please find out, in Azure, how many disk types are billed on actually allocated disk size, and what are those disk types? If you're lack of information, please say don't know.",
        # "As you know, for some disk types, they are billed on actually allocated and used disk size; for other disk types, they are billed on its full bucket size, no matter how much exactly used. Please find out, in Azure, how many disk types are billed on bucket size, and what are those disk types? If you're lack of information, please say don't know.",
        # "can all disks be used to host operating systems for virtual machines?",
        # "how many disk types can be used as OS disk, and what are they?",
        # "how many disk types can not be used as OS disk, and what are they?",
        "how many expanded IOPS per disk does S30 have?",
    ]
    for _q in _qa:
        _re= qa_faiss_vdb_azure(_q)
        print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

    # _qa = [
    #     "what's the difference between Agent and Chain in Langchain?"
    # ]
    # for _q in _qa:
    #     print(_q)
    #     _re= qa_faiss_vdb_langchain(_q)
    #     print(f"\n>>>'{_q}'\n<<<'{_re[0]}'\n")

