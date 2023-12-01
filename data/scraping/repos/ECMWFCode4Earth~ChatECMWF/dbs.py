"""
This module loads the vector databases, currently stored locally. The files have been generated in June 2023, from Confluence, Github and web-content from ECMWF.
Different embedding models have been employed, all based on Sentence Transformers. Three different databases exists, one with the web content, another containing Confluence and Github resources, and
a last one embedding all the Charts API endpoints from the OpenAPI specifications that can be found at [ECMWF](https://charts.ecmwf.int/opencharts-api/v1/search/).
"""
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import DeepLake

from .config import Logger, configs
from .model import llm, memory

embeddings_1 = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_2 = "sentence-transformers/all-mpnet-base-v2"

db_configs = [
    (configs.DS_ECMWF_WEB, embeddings_1),
    (configs.DS_CONFLUENCE_GITHUB, embeddings_2),
    (configs.DS_OPENAPI_REF, embeddings_2),
]
dbs = list()

for config in db_configs:
    embeddings = HuggingFaceHubEmbeddings(repo_id=config[1])
    Logger.info(f"Loading db...{config}")
    db = DeepLake(config[0], embedding_function=embeddings, read_only=True)
    Logger.info(f"Database summary: {db.vectorstore.summary()}")
    dbs.append(db)


ecmwf_web = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=dbs[0].as_retriever(),
    return_source_documents=True,
)

ecmwf_kb = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=dbs[1].as_retriever(),
    return_source_documents=True,
)

openapi_retriever = dbs[2].as_retriever()
openapi_retriever.search_kwargs = {"embedding_function": embeddings}
openapi_ref = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=openapi_retriever,
    return_source_documents=True,
)
