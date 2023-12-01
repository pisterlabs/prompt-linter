import urllib3
urllib3.disable_warnings()

def name2txt(_name):
    _1 = _name.split("_")
    _txt = " ".join(_1)
    return _txt

def clean_txt(_txt):
    import re
    _1 = re.sub(r"\n+", "\n", _txt)
    _2 = re.sub(r"\t+\n", "\n", _1)
    _3 = re.sub(r" +\n", "\n", _2)
    _clean_txt = re.sub(r"\n+", "\n", _3)
    return _clean_txt

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def get_docs_from_links(_links):
    import nest_asyncio
    nest_asyncio.apply()
    from langchain.document_loaders import WebBaseLoader
    with open(_links, 'r') as lf:
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
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    splited_docs = text_splitter.split_documents(_docs)
    return splited_docs

def embedding_to_chroma_ST(_splited_docs, _db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(_splited_docs, embedding_function, persist_directory=_db_name)
    db.persist()
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

def get_chroma_ST(_db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(embedding_function=embedding_function, persist_directory=_db_name)
    return db

def get_chroma_OpenAI(_db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    embedding_function = OpenAIEmbeddings()
    db = Chroma(embedding_function=embedding_function, persist_directory=_db_name)
    return db

def similarity_search_chroma_ST(_query, _db_name):
    db = get_chroma_ST(_db_name)
    similar_docs = db.similarity_search_with_score(_query)
    # similar_docs = db.similarity_search(_query)
    return similar_docs


def get_self_query_retriever(_db_name):
    db = get_chroma_ST(_db_name)
    from langchain.llms import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Web link of the document",
            type="string",
        ),
        AttributeInfo(
            name="title",
            description="Title of the document",
            type="string",
        ),
        AttributeInfo(
            name="description",
            description="Description of the document",
            type="string",
        ),
        AttributeInfo(
            name="language",
            description="Used language in the document",
            type="string",
        ),
    ]
    llm = OpenAI(temperature=0)
    _document_contents = name2txt(_db_name)
    _retriever = SelfQueryRetriever.from_llm(llm, db, _document_contents, metadata_field_info, verbose=True)
    return _retriever

def qa_retriever_self_query(_query, _db_name):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    import os
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _retriever = get_self_query_retriever(_db_name)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=_retriever)
    _ans = qa_chain.run(_query)
    return _ans


_db_name = "../vdb/azure_disk"

# docs = get_docs_from_links('t5.disk.link')
# for doc in docs:
#     doc.page_content = clean_txt(doc.page_content)
#     print(doc.metadata)
# for doc in docs:
#     print(len(doc.page_content))
#     print(doc.metadata['description'])
#     print(doc.metadata['source'],"\n")
#     print(f">>>{doc.page_content}<<<\n")

# splited_docs = split_docs_recursive(docs)
# print(f"splited_docs: {len(splited_docs)}")

# embedding_to_chroma_ST(splited_docs, _db_name)

# _query = "how to save money on disk"
# similar_docs = similarity_search_chroma_ST(_query, _db_name)
# for d in similar_docs:
#     doc = d[0]
#     score = d[1]
#     print(score)
#     print(doc.metadata['description'])
#     print(doc.metadata['source'])
#     print(len(doc.page_content))
#     print(f">>>{doc.page_content}<<<\n")

# ans = qa_retriever_self_query(_query, _db_name)
# print(f"\nans:{ans}\n")

_qa = [
    "how to save moneyr on disk?",
    "how many disk types are billed per actually allocated disk size?",
    "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
    "can all disks be used to host operating systems for virtual machines?",
]
    
for _q in _qa:
    _re= qa_retriever_self_query(_q, _db_name)
    print(_re)

