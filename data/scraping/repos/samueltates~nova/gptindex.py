import logging
import sys
import tempfile
import json
import base64
import os
import asyncio
from human_id import generate_id
from prisma import Json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from session.appHandler import app, websocket
from session.sessionHandler import novaConvo, active_cartridges, cartdigeLookup
from session.prismaHandler import prisma
from core.cartridges import addCartridgeTrigger, update_cartridge_field, addCartridge, addCartridgePrompt
from tools.debug import eZprint
from tools.GoogleDocsReader import GoogleDocsReader 
from tools.UnstructuredReader import UnstructuredReader
from tools.UnstructuredURLLoader import UnstructuredURLLoader

from llama_index import (
    Document,
    # GPTSimpleVectorIndex, 
    GPTListIndex,
    StringIterableReader,
    download_loader,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    StorageContext, load_index_from_storage,
    ServiceContext,
    GPTVectorStoreIndex,
    
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.logger import LlamaLogger
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
# from IPython.display import Markdown, display


llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

#query Index
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform

async def indexDocument(payload, client_loadout):
    eZprint('indexDocument called')
    # print(payload)

    userID = payload['userID']
    indexType = payload['indexType']
    tempKey = payload['tempKey']
    sessionID = payload['sessionID']
    convoID = payload['convoID']
    # sessionID = payload['sessionID']
    document = None
    documentTitle = None
    if payload['document_type'] == 'googleDoc':
        print('google doc triggered')
        gDocID = payload['gDocID']
        loader = GoogleDocsReader() 
        # print(loader)
        # try:
        print('loading data')
        document = await loader.load_data([gDocID], sessionID)
        documentTitle = await loader._get_title(str(gDocID), sessionID)
            # print(document)
        # except:
        #     print('document not found')
        #     # payload = { 'key':tempKey,'fields': {'status': 'doc not found'}}
        #     # cartVal['status'] = 'doc not found'
        #     # input = {
        #     # 'cartKey': tempKey,
        #     # 'convoID': convoID,
        #     # 'fields': {
        #     #     'status': cartVal['status'],
        #     #     },
        #     #     'loadout' : client_loadout
        #     # }
            # await update_cartridge_field(input, client_loadout, system=True)
        # print(document)
        
    elif payload['document_type'] == 'file':
        file_content = payload['file_content']
        file_name = payload['file_name']
        documentTitle = file_name
        file_type = payload['file_type']
        eZprint('reconstructing file')
        payload = { 'key':tempKey,'fields': {'status': 'file recieved, indexing'}}
        await websocket.send(json.dumps({'event':'updateCartridgeFields', 'payload':payload}))
        # print(file_type)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="."+file_type.split('/')[1])
        temp_file.write(file_content)

        # Read and process the reconstructed file
        temp_file.close()
        
        unstructured_reader = UnstructuredReader()
        document = unstructured_reader.load_data(temp_file.name)
    # Cleanup: delete the temporary file after processing
        os.unlink(temp_file.name)

    elif payload['document_type'] == 'url':
        url = payload['url']
        documentTitle = url
        loader = UnstructuredURLLoader([url])
        document = loader.load()
        # print(document)

    index = None
# if(indexType == 'Vector'):
    index = GPTVectorStoreIndex.from_documents(document)
    eZprint('vector index created')
    # if(indexType == 'List'):
    #     index = GPTListIndex.from_documents(document)
    #     eZprint('list index created')

    # tmpfile = tempfile.NamedTemporaryFile(mode='w',delete=False, suffix=".json")
    tmpDir = tempfile.mkdtemp()+"/"+sessionID+"/storage"
    index.storage_context.persist(tmpDir)
    # print(tmpDir)
    indexJson = dict()
    for file in os.listdir(tmpDir):
        if file.endswith(".json"):
            content = json.load(open(os.path.join(tmpDir, file)))
            # print(content)
            indexJson.update({file:content})
  

    key = generate_id()
    index_store = None
    vector_store = None
    docstore = None
    if 'index_store.json' in indexJson:
        index_store = indexJson['index_store.json']
    if 'vector_store.json' in indexJson:
        vector_store = indexJson['vector_store.json']
    if 'docstore.json' in indexJson:
        docstore = indexJson['docstore.json']    

    index_blob =  {
        'key' :key,
            'label': documentTitle,
            'type': 'index',
            'description': 'a document indexed to be queriable by NOVA',
            'enabled': True
    }

    index = await prisma.index.create(

        data = {'key' :key,
                'UserID' : userID,
                'docstore': Json(docstore),
                'index_store': Json(index_store),
                'vector_store': Json(vector_store),
                'blob': Json(index_blob),
        }
    )

  
    cartVal = {
        'label' : documentTitle,
        'type': 'index',
        'description': 'a document indexed to be queriable by NOVA',
        'enabled': True,
        'blocks': {},
        'index': key,
        'status': 'index created, getting summary',
        'position' : 99
    }

    input = {
    'sessionID': sessionID,
    'tempKey': tempKey,
    'newCart': {tempKey:cartVal}
    }
    newCart = await addCartridgePrompt(input, convoID, client_loadout)
    
    return newCart

async def QuickUrlQuery(url, query):
    print('quick url query')
    loader = UnstructuredURLLoader([url])
    document = loader.load()
    index = GPTVectorStoreIndex.from_documents(document)
    response = await queryIndex(query, index)
    return response

async def reconstructIndex(indexJson):
    tmpDir = tempfile.mkdtemp()
    # print(index)
    # nova.eZprint("reconstructIndex: tmpDir={}".format(tmpDir))
    # print(indexJson)
    llama_logger = LlamaLogger()
    service_context = ServiceContext.from_defaults(llama_logger=llama_logger)
    # service_context.set_global_service_context(service_context)
    # print(indexJson)
    for key, val in indexJson.items():
        if key == 'index_store' or key == 'vector_store' or key == 'docstore':
            # print(os.path.join(tmpDir, key))
            # eZprint("reconstructIndex: key={}".format(key))
            # print(val)
            # GPTListIndex.service_context = service_context
            # index = GPTListIndex.build_index_from_nodes(val)
            # print(index)

            try:
                with open(os.path.join(tmpDir, key + '.json'), "w") as f:
                    json.dump(val, f)
            except Exception as e:
                print(f"Error writing file: {str(e)}")

    storage_context = StorageContext.from_defaults(persist_dir=tmpDir, )
    storage_context.persist(tmpDir)
    eZprint("reconstructIndex: storage_context={}".format(storage_context))
    index = load_index_from_storage(storage_context)
   

    return index

    
async def queryIndex(queryString, index ):

    loop = asyncio.get_event_loop()
    query_engine = index.as_query_engine()
    response = await loop.run_in_executor(None, lambda: query_engine.query(queryString))
    eZprint(response)
    
    return response 



async def quick_query(text, query):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    temp_file.write(text.encode())
    temp_file.seek(0)
    unstructured_reader = UnstructuredReader()
    document = unstructured_reader.load_data(temp_file.name)
    index = GPTVectorStoreIndex.from_documents(document)
    response = await queryIndex(query, index)
    return response


def quicker_query(text, query, meta = '' ):
    # print(text)
    document = Document(text, extra_info=meta)
    logger = LlamaLogger()
    logger.set_log_level(logging.DEBUG)
    index = GPTVectorStoreIndex.from_documents([document])
    nodes = index.docstore.get_nodes()
    # print(nodes)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    # print(response)




async def handle_nova_query(cartKey, cartVal, sessionID, query, convoID, loadout = None):
    index_key = cartVal['index']
    indexJson = await get_index_json(index_key)
    index = await reconstructIndex(indexJson)
    insert = await queryIndex(query, index)
    if insert:
        cartVal['blocks'].append(insert)
        cartVal['status'] = 'index query complete'
        
        input = {
        'cartKey': cartKey,
        'sessionID': sessionID,
        'fields': {
            'blocks': cartVal['blocks'],
            'status': cartVal['status']
            },
            'loadout' : loadout
        }
        await update_cartridge_field(input, convoID, loadout, system=True)
    return insert




# async def quicker_query(text, query):
    
#     Document(t) for t in text_list

async def handleIndexQuery(input, convoID, loadout = None):
    cartKey = input['cartKey']
    sessionID = input['sessionID']
    query = input['query']
    cartVal = active_cartridges[convoID][cartKey]
    #TODO -  basically could comine with index query (or this is request, query is internal)

    cartVal['status'] = 'querying Index'
    cartVal['state'] = 'loading'

    input = {
    'cartKey': cartKey,
    'sessionID': sessionID,
    'fields': {
        'status': cartVal['status'],
        'state': cartVal['state']
        },
        'loadout' : loadout
    }
    await update_cartridge_field(input,convoID,loadout, system=True)

    eZprint('handling index query')
    if cartVal['type'] == 'index' and cartVal['enabled'] == True :
        index_key = cartVal['index']
        index = await get_index_json(index_key)
        response = await triggerQueryIndex(sessionID, cartKey, cartVal, query, index, index_key, convoID, loadout )
        return response


async def triggerQueryIndex(sessionID, cartKey, cartVal, query, indexJson, index_key, convoID, loadout = None):
    eZprint('triggering index query')

    cartVal['state'] = 'loading'
    cartVal['status'] = 'index Found'
    input = {
    'cartKey': cartKey,
    'sessionID': sessionID,
    'fields': {
        'status': cartVal['status'],
        'state': cartVal['state']
        },
        'loadout' : loadout
    }
    await update_cartridge_field(input, convoID,  loadout, system=True)
    localIndex = await reconstructIndex(indexJson)
    insert = await queryIndex(query, localIndex)

    indexJson = dict()  

    tmpDir = tempfile.mkdtemp()+"/"+sessionID+"/storage"
    localIndex.storage_context.persist(tmpDir)
    # print(tmpDir)
    indexJson = dict()
    for file in os.listdir(tmpDir):
        if file.endswith(".json"):
            content = json.load(open(os.path.join(tmpDir, file)))
            # print(content)
            indexJson.update({file:content})


    index_store = None
    vector_store = None
    docstore = None

    if 'index_store.json' in indexJson:
        index_store = indexJson['index_store.json']
    if 'vector_store.json' in indexJson:
        vector_store = indexJson['vector_store.json']
    if 'docstore.json' in indexJson:
        docstore = indexJson['docstore.json']   

    print(index_key)
    remote_index = await prisma.index.find_first(
        where={
                'key': index_key
                }
    )
    print('remote found')
    # print(remote_index)
    # if remote_index :
    #     update_index = await prisma.index.update(
    #         where={
    #                 'id': remote_index.id
    #         },
    #         data = {
    #                 # 'docstore': Json(docstore),
    #                 # 'index_store': Json(index_store),
    #                 'vector_store': Json(vector_store),
    #         }
    #     )

    
    eZprint('index query complete')
    # eZprint(insert)
    if(insert != None):
        print('inserting')
        #TODO - replace this ID lookup with a key lookup
        cartVal['state'] = ''
        cartVal['status'] = ''
        # print(cartVal)
        if 'blocks' not in cartVal:
            cartVal['blocks'] = {}
        if 'queries' not in cartVal['blocks']:
            cartVal['blocks']['queries'] = []
        cartVal['blocks']['queries'].append({'query':query, 'response':str(insert)})
        payload = { 'key':cartKey,'fields': {
                            'status': cartVal['status'],
                            'blocks':cartVal['blocks'],
                            'state': cartVal['state']
                                }, 
                                'loadout':loadout
                                }
        input = {
        'cartKey': cartKey,
        'sessionID': sessionID,
        'fields': {
            'status': cartVal['status'],
            'state': cartVal['state'],
            'blocks': cartVal['blocks']
            },
            'loadout' : loadout
        }
        await update_cartridge_field(input, convoID, loadout, system=True)
        return insert

        
async def get_index_json(index_key):
    eZprint('getting cartridge detail')
    print(index_key)
    matchedCart = await prisma.index.find_first(
        where={
                'key': index_key
                }
    )
    # print(matchedCart)
    dbRecord = json.loads(matchedCart.json())

    return dbRecord



def run_gpt_headless():
    text = ''
    query = ''
    query_active = True
    while query_active:
        if text == '':
            text = input('text: ')
            print(text)
        elif query == '':
            query = input('query: ')
            print(query)
        else:
            print('running query')
            response = quicker_query(text, query)
            print(response)
            text = ''
            query = ''
        
        print('done')
            
            



def main() -> None:
    run_gpt_headless()

if __name__ == '__main__':
    # asyncio.run(main())
    main()