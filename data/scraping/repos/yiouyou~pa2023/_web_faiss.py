from _web import txt2name, clean_txt, get_docs_from_links, split_docs_recursive

def embedding_to_faiss_ST(_docs, _db_name):
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2") # all-mpnet-base-v2/all-MiniLM-L6-v2/all-MiniLM-L12-v2
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

def weblinks_to_faiss(_links, _db_name):
    docs = get_docs_from_links(_links)
    if len(docs) > 0:
        for doc in docs:
            doc.page_content = clean_txt(doc.page_content)
            # print(doc.metadata)
        print(f"docs: {len(docs)}")
        splited_docs = split_docs_recursive(docs)
        print(f"splited_docs: {len(splited_docs)}")
        embedding_to_faiss_ST(splited_docs, _db_name)
    else:
        print("NO docs")

def url_recursive_to_faiss(_url, _exclude, _db_name):
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
        embedding_to_faiss_ST(splited_docs, _db_name)
    else:
        print("NO docs")

def weblinks_to_link_md_azure(_links, _dir):
    import re, os, json
    if not os.path.exists(_dir):
        os.makedirs(_dir)
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
        fn = os.path.join(_dir, f"{str(n).zfill(4)}.md")
        print(fn)
        with open(fn, "w") as wf:
            wf.write(_t8)
        link_md[fn] = i
    # print(link_md)
    fn = os.path.join(_dir, "_link_md.json")
    with open(fn, "w", encoding="utf-8") as wf:
        wf.write(json.dumps(link_md, ensure_ascii=False, indent=4))

def weblinks_to_link_md_gmzy(_links, _dir):
    import re, os, json
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    import requests
    from markdownify import markdownify
    with open(_links, "r") as lf:
        _list = lf.read().splitlines()
    n = 0
    link_md = {}
    for i in _list:
        print(i)
        _r = requests.get(i)
        _r_text = _r.text.encode('latin1').decode('utf-8')
        _t1 = markdownify(_r_text, heading_style="ATX")
        _t2 = _t1.split(" twikoo.init")
        _t3 = _t2[0]
        _t4 = re.sub(r'\n\s*\n', '\n\n', _t3)
        _t5 = re.sub(r'\n\:root.+\n\n\\_\\_md.+\nwindow.+\n', '', _t4)
        _t6 = _t5.split("\n# ")
        if len(_t6) > 1:
            if _t6[1]:
                n += 1
                _t7 = "# " + _t6[1]
                fn = os.path.join(_dir, f"{str(n).zfill(4)}.md")
                print(fn)
                with open(fn, "w") as wf:
                    wf.write(_t7)
                link_md[fn] = i
    # print(link_md)
    fn = os.path.join(_dir, "_link_md.json")
    with open(fn, "w", encoding="utf-8") as wf:
        wf.write(json.dumps(link_md, ensure_ascii=False, indent=4))
# """
# 编者 - 本草备要讲解-光明中医教材

# :root{--md-text-font:"Roboto";--md-code-font:"Roboto Mono"}

# \_\_md\_scope=new URL(".",location),\_\_md\_get=(e,\_=localStorage,t=\_\_md\_scope)=>JSON.parse(\_.getItem(t.pathname+"."+e)),\_\_md\_set=(e,\_,t=localStorage,a=\_\_md\_scope)=>{try{t.setItem(a.pathname+"."+e,JSON.stringify(\_))}catch(e){}}
# window.ga=window.ga||function(){(ga.q=ga.q||[]).push(arguments)},ga.l=+new Date,ga("create","UA-151330658-1","auto"),ga("set","anonymizeIp",!0),ga("send","pageview"),document.addEventListener("DOMContentLoaded",function(){document.forms.search&&document.forms.search.query.addEventListener("blur",function(){var e;this.value&&(e=document.location.pathname,ga("send","pageview",e+"?q="+this.value))}),"undefined"!=typeof location$&&location$.subscribe(function(e){ga("send","pageview",e.pathname)})})

# [跳转至](#_1) 
# """

def link_md_to_faiss(_dir, _db_name):
    import os, json
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        # ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    _docs = []
    fn = os.path.join(_dir, "_link_md.json")
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
    embedding_to_faiss_ST(_docs, _db_name)

def link_md_to_faiss_gmzy(_dir, _db_name):
    import os, json
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    _docs = []
    fn = os.path.join(_dir, "_link_md.json")
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
    embedding_to_faiss_ST(_docs, _db_name)


if __name__ == "__main__":

    from _web import txt2name, clean_txt, get_docs_from_links, split_docs_recursive
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _faiss_path = _pwd.parent.parent.parent

    # _db_gmzy = txt2name("gmzy_bak")
    # print(_db_gmzy)
    # _links = str(_faiss_path / "vdb" / "gmzy.link")
    # _gmzy = str(_faiss_path / "vdb" / _db_gmzy)
    # _md_dir = "./md_gmzy_bak"
    # # weblinks_to_link_md_gmzy(_links, _md_dir)
    # link_md_to_faiss_gmzy(_md_dir, _gmzy)

    _db_azure = txt2name("Azure Cache Redis")
    print(_db_azure)
    _links = str(_faiss_path / "vdb" / "azure_cache_redis.link")
    _azure = str(_faiss_path / "vdb" / _db_azure)
    _md_dir = "./md_azure_cache_redis"
    weblinks_to_link_md_azure(_links, _md_dir)
    link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Databricks")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_databricks.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_databricks"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Well-Architected Framework")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_well-architected-framework.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_well-architected_framework"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure VM")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_vm.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_vm"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure SQL DB")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_sql_db.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_sql_db"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure SQL MI")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_sql_mi.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_sql_mi"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure App Service")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_app_service.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_app_service"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Monitor")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_monitor.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_monitor"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Synapse")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_synapse.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_synapse"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Blob Storage")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_blob_storage.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_blob_storage"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_azure = txt2name("Azure Cosmos DB")
    # print(_db_azure)
    # _links = str(_faiss_path / "vdb" / "azure_cosmos_db.link")
    # _azure = str(_faiss_path / "vdb" / _db_azure)
    # _md_dir = "./md_azure_cosmos_db"
    # weblinks_to_link_md_azure(_links, _md_dir)
    # link_md_to_faiss(_md_dir, _azure)

    # _db_langchain = txt2name("Langchain Python Documents")
    # print(_db_langchain)
    # _url = 'https://python.langchain.com/docs/modules/'
    # _exclude = [
    #     'https://python.langchain.com/docs/additional_resources',
    #     'https://api.python.langchain.com/en/latest/',
    # ]
    # _langchain = str(_faiss_path / "vdb" / _db_langchain)
    # url_recursive_to_faiss(_url, _exclude, _langchain)

