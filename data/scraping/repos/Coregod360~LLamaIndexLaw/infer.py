import os
import sys
import shutil
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import langchain
from llama_index import StorageContext, load_index_from_storage
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

def init(resources):
    db = resources['db']
    dir = os.path.dirname(__file__)
    parentDir = os.path.dirname(dir)
    textS3Bucket = 'lawlet-uscode'
    vectorS3Bucket = 'lawlet-uscode-vectors'
    indexS3Bucket = 'lawlet-uscode-index'
    localVectorDir = db
    localSQLDumpDir = dir +'/'+'uscode'
    vectorDownloadDir = dir +'/'+'vectordownload'
    targzFile = localVectorDir + '/uscode.tar.gz'
    if not os.path.exists(localSQLDumpDir):
        os.makedirs(localSQLDumpDir)
        os.system('tar -xvzf ' + targzFile + ' -C ' + dir)
    #if not os.path.exists(localVectorDir):
    #    os.makedirs(localVectorDir)
    #    os.system('copy ' + vectorDownloadDir + '/docstore.json ' + localVectorDir + '/docstore.json')
    #    os.system('copy ' + vectorDownloadDir + '/vector_store.json ' + localVectorDir + '/vector_store.json')
    #    os.system('copy ' + vectorDownloadDir + '/index_store.json ' + localVectorDir + '/index_store.bin')
    #    os.system('copy ' + vectorDownloadDir + '/graph_store.json ' + localVectorDir + '/graph_store.json')
    documents = SimpleDirectoryReader(localSQLDumpDir).load_data()
    print("Rebuild StorageContext")
    storage_context = StorageContext.from_defaults(persist_dir=localVectorDir)
    print("Load index")
    index = load_index_from_storage(storage_context)
    # build index
    #index = VectorStoreIndex.from_documents(documents)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.0)
        ]
    )
    return query_engine

def query(query_engine, inputText):
    queryText = "ignore the question and summarize the retrieved law. Question: " + inputText
    response = query_engine.query(queryText)
    responseText = response.response
    metadata = response.metadata
    metadataKeys = list(metadata.keys())
    node = metadataKeys[0]
    filename = metadata[node]['filename']
    s3filename = metadata[node]['S3filename']
    title = metadata[node]['title']
    number = metadata[node]['number']
    name = metadata[node]['name']
    source_nodes = response.source_nodes
    source_node = source_nodes[0]
    score = source_node.score
    data = None
    if os.path.exists(filename):
        data = open(filename, 'r').read()
    #else:
    #    print("Downloading " + filename + " from S3")
    #    command = 's3cmd get ' + s3filename + ' ' + filename
    #    print(command)
    #    os.system(command)
    #    data = open( filename, 'r').read()
        
    if data == None:
        print("Error: data is None")
        return None 
        
    results = {
        "filename": filename,
        "s3filename": s3filename,
        "name": name,
        "title": title,
        "nr": number,
        "score": score,
        "summary": responseText,
        "body": data
    }
    return results


if __name__ == '__main__':
    question = "What does the law about Time of election say?"
    queryEngine = init()
    queryResult = query(queryEngine, question)
    print(queryResult)
