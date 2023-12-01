from _web import txt2name, clean_txt, get_docs_from_links, split_docs_recursive

def embedding_to_chroma_ST(_docs, _db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    _embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # all-mpnet-base-v2
    if len(_docs) > 0:
        print(0)
        db = Chroma.from_documents([_docs[0]], _embedding_function, persist_directory=_db_name)
        db.persist()
    if len(_docs) > 1:
        for i in range(1, len(_docs)):
            print(i)
            _db = Chroma(embedding_function=_embedding_function, persist_directory=_db_name)
            _db.add_documents([_docs[i]])
            _db.persist()
    print(_db_name)
    print("[chroma save HuggingFaceEmbeddings embedding to disk]")

def embedding_to_chroma_OpenAI(_splited_docs, _db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(_splited_docs, embedding_function, persist_directory=_db_name)
    db.persist()
    print("[chroma save OpenAI embedding to disk]")

def weblinks_to_chroma(_links, _db_name):
    docs = get_docs_from_links(_links)
    if len(docs) > 0:
        for doc in docs:
            doc.page_content = clean_txt(doc.page_content)
            # print(doc.metadata)
        print(f"docs: {len(docs)}")
        splited_docs = split_docs_recursive(docs)
        print(f"splited_docs: {len(splited_docs)}")
        embedding_to_chroma_ST(splited_docs, _db_name)
    else:
        print("NO docs")

def weblinks_to_md_to_chroma(_links, _db_name):
    import re
    import requests
    from markdownify import markdownify
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    with open(_links, "r") as lf:
        _list = lf.read().splitlines()
    _docs = []
    for i in _list:
        print(i)
        _r = requests.get(i)
        _t1 = markdownify(_r.text, heading_style="ATX")
        _t2 = re.sub(r'\n\s*\n', '\n\n', _t1)
        i_md = markdown_splitter.split_text(_t2)
        for j in i_md:
            _m1 = j.metadata
            _headers = _m1.values()
            _m1['description'] = ', '.join(_headers)
            _m1['source'] = i
            j.metadata = _m1
        _docs += i_md
    print(f"list: {len(_list)}")
    print(f"docs: {len(_docs)}")
    print(_docs[-1])
    embedding_to_chroma_ST(_docs, _db_name)

def weblinks_to_md(_links, _md_dir):
    import re, os, json
    import requests
    from markdownify import markdownify
    with open(_links, "r") as lf:
        _list = lf.read().splitlines()
    n = 0
    link_md = {}
    for i in _list:
        n += 1
        print(i)
        _r = requests.get(i)
        _t1 = markdownify(_r.text, heading_style="ATX")
        _t2 = re.sub(r'\n\s*\n', '\n\n', _t1)
        _t3 = _t2.split("\nTable of contents\n\n")
        _t4 = _t3[-1]
        _t5 = _t4.split("\n## Additional resources\n\n")
        _t6 = _t5[0]
        _t7 = _t6.split("\nTheme\n\n")
        _t8 = _t7[0]
        fn = os.path.join(_md_dir, f"{str(n).zfill(3)}.md")
        print(fn)
        with open(fn, "w") as wf:
            wf.write(_t8)
        link_md[fn] = i
    # print(link_md)
    fn = os.path.join(_md_dir, "link_md.json")
    with open(fn, "w", encoding="utf-8") as wf:
        wf.write(json.dumps(link_md, ensure_ascii=False, indent=4))

def md_to_chroma(_md_dir, _db_name):
    import os, json
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    _docs = []
    fn = os.path.join(_md_dir, "link_md.json")
    with open(fn, "r", encoding="utf-8") as rf:
        link_md = json.loads(rf.read())
    # print(link_md)
    for i in link_md:
        print(i)
        with open(i, "r", encoding="utf-8") as rf:
            _t = rf.read()
            _md = markdown_splitter.split_text(_t)
            for j in _md:
                _m1 = j.metadata
                _headers = _m1.values()
                _m1['description'] = ', '.join(_headers)
                _m1['source'] = link_md[i]
                j.metadata = _m1
            _docs += _md
    print(f"docs: {len(_docs)}")
    embedding_to_chroma_ST(_docs, _db_name)

def url_recursive_to_chroma(_url, _exclude, _db_name):
    import nest_asyncio
    nest_asyncio.apply()
    from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
    loader = RecursiveUrlLoader(url=_url, exclude_dirs=_exclude)
    loader.verify = False
    loader.requests_per_second = 1
    docs=loader.load()
    if len(docs) > 0:
        for doc in docs:
            doc.page_content = clean_txt(doc.page_content)
            print(doc.metadata)
        print(f"docs: {len(docs)}")
        splited_docs = split_docs_recursive(docs)
        print(f"splited_docs: {len(splited_docs)}")
        embedding_to_chroma_ST(splited_docs, _db_name)
    else:
        print("NO docs")

def url_tables_to_chroma(_url, _db_name):
    docs = get_tables_from_url(_url)
    if len(docs) > 0:
        print(f"docs: {len(docs)}")
        splited_docs = split_docs_recursive(docs)
        print(f"splited_docs: {len(splited_docs)}")
        embedding_to_chroma_ST(splited_docs, _db_name)
    else:
        print("NO docs")

def get_tables_from_url(_url):
    import requests
    from bs4 import BeautifulSoup as bs
    import pandas as pd
    from pprint import pprint
    from langchain.schema import Document
    _docs = []
    _r = requests.get(_url)
    _soup = bs(_r.text, 'html.parser')
    _t0 = _soup.find('table')
    if _t0:
        _is_table = True
    else:
        _is_table = False
    while _is_table:
        _p0 = _t0.find_previous_sibling('p').get_text()
        print(_p0)
        _d0 = pd.read_html(_t0.prettify())
        # print(type(_d0[0].to_string()))
        print(_d0[0].to_string())
        # _d0[0].to_csv(f"{_p0}.csv", index = False)
        _source = f"{_p0} - {_url}"
        print(f"\nsource:'{_source}'\n")
        _docs.append(
            Document(
                page_content=_d0[0].to_string(),
                metadata={
                    "source": _source,
                    "description": _p0,
                }
            )
        )
        _t1 = _t0.find_next('table')
        if _t1:
            _is_table = True
            _t0 = _t1
        else:
            _is_table = False
    return _docs


if __name__ == "__main__":

    from _web import txt2name, clean_txt, get_docs_from_links, split_docs_recursive
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _chroma_path = _pwd.parent.parent.parent

    # _db_azure = txt2name("Azure disks introduction")
    # print(_db_azure)
    # _links = str(_chroma_path / "vdb" / "azure_disk.link")
    # _azure = str(_chroma_path / "vdb" / _db_azure)
    # weblinks_to_md_to_chroma(_links, _azure)

    # _db_azure = txt2name("Introduction to Azure managed disks")
    # print(_db_azure)
    # _links = str(_chroma_path / "vdb" / "azure_disk.link")
    # _azure = str(_chroma_path / "vdb" / _db_azure)
    # weblinks_to_chroma(_links, _azure)

    # _db_langchain = txt2name("Langchain Python Documents")
    # print(_db_langchain)
    # _url = 'https://python.langchain.com/docs/modules/'
    # _exclude = [
    #     'https://python.langchain.com/docs/additional_resources',
    #     'https://api.python.langchain.com/en/latest/',
    # ]
    # _langchain = str(_chroma_path / "vdb" / _db_langchain)
    # url_recursive_to_chroma(_url, _exclude, _langchain)

    # _db_azure_tables = txt2name("Tables of Azure managed disks")
    # print(_db_azure_tables)
    # _url = "https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types"
    # _azure_tables = str(_chroma_path / "vdb" / _db_azure_tables)
    # print(_azure_tables)
    # url_tables_to_chroma(_url, _azure_tables)

