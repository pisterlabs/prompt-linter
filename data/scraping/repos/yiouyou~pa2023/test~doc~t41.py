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


def get_chroma_ST(_db_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

# docs = get_docs_from_links('../vdb/t41.link')
# for doc in docs:
#     print(len(doc.page_content))
#     doc.page_content = clean_txt(doc.page_content)
#     print(doc.metadata)
#     print(len(doc.page_content))
    # print(doc.page_content)

# import pandas as pd
# df = pd.read_html('https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types')
# print(len(df))
# for i in range(len(df)):
#     # df[i].rename(columns = {'Unnamed: 0':'name'}, inplace = True)
#     df[i].to_csv(f"t41_{i+1}.csv", index = False)

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()
# agent = create_csv_agent(
#     OpenAI(temperature=0),
#     "t41_1.csv",
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )
from langchain.chat_models import ChatOpenAI
agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    "1688285921_5.csv",
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
_re = agent.run("how many expanded IOPS per disk does S30 have?")
print(_re)

# _db_azure = "../vdb/introduction_to_azure_managed_disks"
# _db_langchain = "../vdb/langchain_python_documents"

# _qa = [
#     "how to save moneyr on disk?",
#     "how many disk types are billed per actually allocated disk size?",
#     "how many disk types are billed per actually allocated disk size and how many is billed in buckets?",
#     "can all disks be used to host operating systems for virtual machines?",
# ]
# _query = "how many disk types are billed per actually allocated disk size?"
# print(_query)
# similar_docs = similarity_search_chroma_ST(_query, _db_azure)
# for d in similar_docs:
#     doc = d[0]
#     score = d[1]
#     print("\n", score)
#     print(f"title: {doc.metadata['title']}")
#     print(f"descr: {doc.metadata['description']}")
#     print(f"title: {doc.metadata['source']}")
#     print(len(doc.page_content))
#     # print(f">>>{doc.page_content}<<<\n")
# ans = qa_retriever_self_query(_query, _db_azure)
# print(ans)



# _query = "how to write a langchain agent?"
# print(_query)
# similar_docs = similarity_search_chroma_ST(_query, _db_langchain)
# for d in similar_docs:
#     doc = d[0]
#     score = d[1]
#     print("\n", score)
#     print(f"title: {doc.metadata['title']}")
#     print(f"description: {doc.metadata['description']}")
#     print(f"source: {doc.metadata['source']}")
#     print(len(doc.page_content))
#     print(f">>>{doc.page_content}<<<\n")
# ans = qa_retriever_self_query(_query, _db_langchain)
# print(ans)


# from langchain.document_loaders import BSHTMLLoader
# # loader = BSHTMLLoader("https://learn.microsoft.com/en-us/azure/cost-management-billing/reservations/understand-disk-reservations?toc=/azure/virtual-machines/toc.json")
# loader = BSHTMLLoader("https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types")
# data = loader.load()
# print(data)
