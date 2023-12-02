from langchain.document_loaders import DirectoryLoader,PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Milvus
from pathlib import Path
from config import config_ns
import os

MILVUS_DB_HOST=os.getenv("MILVUS_DB_HOST")
print("MILVUS_DB_HOST")
model_name = Path(Path(__file__).resolve().parents[1],f"embedding_model/{config_ns.retriever['emb_model_name']}").as_posix()
model_kwargs = {'device': config_ns.retriever['device_type']}
encode_kwargs = {'normalize_embeddings': True}

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 10,
        length_function = len,
        is_separator_regex = False,
    )


# TODO: Create a singelton class for embedding_model
# used in two places: data_loader and retriever
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def pipeline(path:str):
    # DB
    # can configure collection_name, drop_old, connection_args = {}
    vector_db = Milvus(embedding_function=embedding_model,collection_name='LangChainCollection',drop_old=True,
                        connection_args={'host':MILVUS_DB_HOST})
    
    # Currently we support two loaders: pdf loader which uses pypdf and text loader
    loaders = {"pdf_loader":PyPDFDirectoryLoader(path),
                "text_loader":DirectoryLoader(path=path,glob="**/*.txt",
                                            show_progress=True,loader_cls=TextLoader)}
    docs = []
    for loader in loaders.values():
        docs.extend(loader.load())  
    # TODO Log number of docs In pdf each page is treated as a separate document.

    # Make content and metadata
    content = [i.page_content for i in docs]
    metadata = [{'source':i.metadata['source']} for i in docs]

    # Split
    chunks = text_splitter.create_documents(texts=content,metadatas=metadata)
    # TODO log length of chunks created

    # Index into DB
    index = vector_db.add_documents(chunks)  #Returns the pk of added documents
    # TODO log length of index
    return len(docs),len(index)
