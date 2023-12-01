import os
import sys
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from llama_index import Document , ListIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index import LLMPredictor, VectorStoreIndex, ServiceContext
from langchain import OpenAI
from llama_index import StorageContext, load_index_from_storage
import random
import json
storage_context = StorageContext.from_defaults()
parser = SimpleNodeParser()

def main(filtered, metadata):
    dir = os.path.dirname(__file__)
    parentDir = os.path.dirname(dir)
    # randomize order of filtered and metadata
    random.shuffle(filtered)
    random.shuffle(metadata)
    # only 10 documents for testing
    #filtered = filtered[:5000]
    #metadata = metadata[:5000]
    print("load")
    chunkSize = 1024
    textS3Bucket = 'lawlet-uscode'
    vectorS3Bucket = 'lawlet-uscode-vectors'
    localVectorDir = parentDir +'/'+'vectorstore'
    localSQLDumpDir = './'+'uscode'
    allChunks = []
    documents = []
    if not os.path.exists(localSQLDumpDir):
        os.makedirs(localSQLDumpDir)
    if not os.path.exists(localVectorDir):
        os.makedirs(localVectorDir)
    listSqlDump = os.listdir(localSQLDumpDir)
    if len(listSqlDump) > 0:
        for file in listSqlDump:
            os.remove(localSQLDumpDir+'/'+ file)
    if os.path.exists(localVectorDir + '/docstore.json'):
        storage_context = StorageContext.from_defaults(persist_dir=localVectorDir)
        index = load_index_from_storage(storage_context)
    else:
        storage_context = StorageContext.from_defaults()
    for i in range(len(filtered)):
        thisMetadata = metadata[i]
        row = filtered[i]
        firstLine = row.split("\n")[0]
        body = row.split("\n")[1]
        concat = firstLine + body
        localfilename = localSQLDumpDir+'/'+ thisMetadata[2] + "-" + thisMetadata[0] + '.txt'
        S3filename = 's3://'+ textS3Bucket +'/' + thisMetadata[2] + "-" + thisMetadata[0] + '.txt'
        with open(localfilename, 'a') as f:
            f.write(concat)
        documents.append(
            Document(
                text=concat,
                metadata={
                    'filename': localfilename,
                    'S3filename': S3filename,
                    'title': thisMetadata[2],
                    'number': thisMetadata[0],
                    'name': thisMetadata[1]
                }
            )
        )
    #print("adding documents to textS3Bucket")
    #command = 's3cmd put '+ sqlDumpDir + '/* ' + "s3://"+ textS3Bucket +"/ --recursive"
    #print(command)
    #os.system(command)
    print('skipping adding documents to textS3Bucket')
    print("adding documents to nodes")
    nodes = parser.get_nodes_from_documents(documents)
    print("adding nodes to docstore")
    storage_context.docstore.add_documents(nodes)
    print("adding nodes to vectorstore")
    thisVectorIndex = VectorStoreIndex(nodes, storage_context=storage_context)
    print("persisting vectorstore")
    thisVectorIndex.storage_context.persist(persist_dir=localVectorDir)
    print("adding nodes to listindex")
    os.system('s3cmd put '+ localVectorDir + ' ' + "s3://"+ vectorS3Bucket +"/ --recursive")
if __name__ == '__main__':
    main()