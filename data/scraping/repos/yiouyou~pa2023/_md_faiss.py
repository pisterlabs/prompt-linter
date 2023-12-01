from _md import txt2name, clean_txt, get_docs_from_links, split_docs_recursive

def embedding_to_faiss_ST(_docs, _db_name):
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # all-mpnet-base-v2
    _db = FAISS.from_documents(_docs, _embeddings)
    _db.save_local(_db_name)
    print(_db_name)
    print("[faiss save HuggingFaceEmbeddings embedding to disk]")

def embedding_to_faiss_OpenAI(_docs, _db_name):
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    _embeddings = OpenAIEmbeddings()
    _db = FAISS.from_documents(_docs, _embeddings)
    _db.save_local(_db_name)
    print("[faiss save OpenAI embedding to disk]")

def md_dir_to_faiss(_dir, _db_name):
    import os
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    _md_list = []
    for (root, dirs, files) in os.walk(_dir, topdown=True):
        for name in files:
            _f = os.path.join(root, name)
            if '.md' in _f:
                _md_list.append(_f)
        for name in dirs:
            _f = os.path.join(root, name)
            if '.md' in _f:
                _md_list.append(_f)
    _docs = []
    for i in _md_list:
        print(i)
        head_tail = os.path.split(i)
        with open(i, "r", encoding="utf-8") as rf:
            _t = rf.read()
            _md = markdown_splitter.split_text(_t)
            for j in _md:
                _m1 = j.metadata
                _headers = _m1.values()
                _m1['description'] = ', '.join(_headers)
                _m1['source'] = head_tail[1]
                _m1['title'] = head_tail[1].replace(".md", "").replace("-", " ")
                j.metadata = _m1
            _docs += _md
    print(f"docs: {len(_docs)}")
    print(_docs[-1])
    embedding_to_faiss_ST(_docs, _db_name)


if __name__ == "__main__":

    from _md import txt2name, clean_txt, get_docs_from_links, split_docs_recursive
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _faiss_path = _pwd.parent.parent.parent

    # _db_azure_vm = txt2name("Azure Virtual Machines")
    # print(_db_azure_vm)
    # _azure_vm = str(_faiss_path / "vdb" / _db_azure_vm)
    # md_dir_to_faiss("./vm", _azure_vm)

