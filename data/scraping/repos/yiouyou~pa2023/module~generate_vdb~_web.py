def txt2name(_txt):
    import re
    _1 = re.sub(r" +", " ", _txt)
    _2 = _1.lower()
    _3 = _2.split(" ")
    _name = "_".join(_3)
    return _name

def clean_txt(_txt):
    import re
    _1 = re.sub(r"\n+", "\n", _txt)
    _2 = re.sub(r"\t+\n", "\n", _1)
    _3 = re.sub(r" +\n", "\n", _2)
    _clean_txt = re.sub(r"\n+", "\n", _3)
    return _clean_txt

def get_docs_from_links(_links):
    import nest_asyncio
    nest_asyncio.apply()
    from langchain.document_loaders import WebBaseLoader
    with open(_links, "r") as lf:
        _list = lf.read().splitlines()
    print(len(_list))
    loader = WebBaseLoader(_list)
    loader.verify = False
    loader.requests_per_second = 1
    docs = loader.load()
    return docs

def split_docs_recursive(_docs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 4000, #1000,
        chunk_overlap = 200, #200,
        length_function = len,
    )
    splited_docs = text_splitter.split_documents(_docs)
    return splited_docs

