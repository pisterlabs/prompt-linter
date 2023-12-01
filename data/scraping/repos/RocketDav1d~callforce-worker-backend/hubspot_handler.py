import asyncio
import aiohttp
import re
import os
import openai
import hashlib
import time
import dotenv
import chromadb
from typing import Dict
from hubspot import HubSpot
import time
import sys
from deepgram import Deepgram
import asyncio, json
from translate import Translator


dotenv.load_dotenv()



# # Initialize Hubspot
# hubspot_access_token = "CPuu4Yq4MRIMAAEAQAAAAQIAAAA4GPrFmkQg1OivHCje94wBMhS5CJjYBts5VUPQlIni2oMerXhVQzowAAAAQQAAAAAAAAAAAAAAAACAAAAAAAAAAAAAIAAAAA4A4AEAAAAAAAD8BwAAAPADQhQLholqVPqTga-k9SuBHIgcwvBGmUoDZXUxUgBaAA"
# hubspot = HubSpot(access_token=hubspot_access_token)



# Intialize ChromaDb
# chroma_client = HttpClient(host="localhost", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
# chroma_client = chromadb.HttpClient(host="http://18.197.143.82", port = 8000)
# chroma_client = chromadb.Client()
# print("ChromaDb Client: ", chroma_client


async def initialize_chroma_client():
    chroma_client = chromadb.HttpClient(host="18.197.143.82", port=8000)
    print("Successfully initialized ChromaDb client", chroma_client)
    return chroma_client


async def get_client(hubspot_access_token):
    hubspot = HubSpot(access_token=hubspot_access_token)
    print("successfully created hubspot client")
    return hubspot


async def get_embedding(text_to_embed):
    # print(f"Embedding text: {text_to_embed}")
    try:
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=text_to_embed
        )
        embedding = response["data"][0]["embedding"]
        print(f"Successfully embedded text: {text_to_embed}")
        return embedding
    except openai.error.InvalidRequestError as e:
        print(f"Skipping embedding for invalid text: {text_to_embed}")
        print(f"Error: {str(e)}")
        return None



async def fetch_properties(session, object_type, access_token):
    url = f'https://api.hubapi.com/crm/v3/properties/{object_type}'
    headers = {
        'authorization': "Bearer " + access_token,
    }
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                # print(data)
                # return data
                results = data.get('results', [])
                properties = []
                for prop in results:
                    name = prop.get('name', '')
                    print(name)
                    description = prop.get('description', '')
                    # embedding_name = await get_embedding(name)
                    # embedding_description = await get_embedding(description)
                    property_dict = {
                        "name": name,
                        "description": description,
                        # "embedding_name": embedding_name,
                        # "embedding_description": embedding_description,
                        "label": prop.get('label'),
                        "hidden": prop.get('hidden'),
                        "archived": prop.get('archived'),
                        "object_type": object_type
                    }
                    properties.append(property_dict)
                # print(properties)
                return {object_type: properties}
            else:
                return {object_type: f"Failed to fetch properties, status code: {response.status}"}
    except Exception as e:
        print(f"Exception when calling HubSpot API: {str(e)}")
        return {object_type: "Failed to fetch properties"}
    




async def return_properties(access_token):
    async with aiohttp.ClientSession() as session:
        types = ["contacts", "companies", "deals"]
        tasks = [fetch_properties(session, object_type, access_token) for object_type in types]
        properties = await asyncio.gather(*tasks)
        print("PROPERTIES: ", properties)
        return properties
        # await save_properties_to_chroma({k: v for d in properties for k, v in d.items()})






async def save_properties_to_chroma(properties, file_key, userId):
    collection_name = userId + "_properties"
    chroma_client = await initialize_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for prop in properties:
    # Assuming 'embedding_name' and 'embedding_description' are keys in prop
        if 'embedding_name' in prop and 'embedding_description' in prop:
            embedding = prop["embedding_name"] + prop["embedding_description"]
        else:
            print(f"Skipping property due to missing embedding data: {prop}")
            continue
        metadata = { 
            "name": prop["name"],
            "label": prop["label"],
            "description": prop["description"],
            "type": prop["type"],
            "userId": prop["userId"],
            "file_key": file_key
        }
        print("add to chroma")
        await collection.add(embeddings=[embedding], metadatas=[metadata])



async def query_the_collection(file_key, userId, properties):
    print("QUERY THE COLLECTION")
    chroma_client = await initialize_chroma_client()
    collection = chroma_client.get_or_create_collection(name=userId)
    translator= Translator(to_lang="de")

    # Asynchronously gather all query results
    query_tasks = []
    for prop in properties:
        german_name = translator.translate(prop.label) 
        print(german_name) # Accessing name using dot notation
        query_task = query_property(collection, german_name, file_key)
        print(query_task)
        query_tasks.append(query_task)

    # Wait for all tasks to complete
    query_results = await asyncio.gather(*query_tasks)
    return query_results

async def query_property(collection, property_name, file_key):
    embedding = await get_embedding(property_name)
    query_result = collection.query(
        query_embeddings=embedding,
        n_results=6,
        where={"document": file_key},
    )
    return {"query_term": property_name, "result": query_result}













#def vectorize_query(query_input, vectordb_client, collection_name, chat_id):
#     embeddings = get_embedding(query_input)
#     collection = vectordb_client.get_or_create_collection(name=collection_name)
#     metadata = {"query": query_input, "document": chat_id}
#     id = hashlib.md5(query_input.encode()).hexdigest()
#     collection.add(documents=query_input, embeddings=embeddings, metadatas=metadata, ids=id)




# while True:
#   query=input(f"Prompt:")
#   if query == "exit":
#     print('Exiting')
#     sys.exit()
#   if query == '':
#     continue
#   result = pdf_qa({"question": query})
#   print(f"Answer: " + result["answer"])





















# async def fetch_properties(session, object_type):  
#     url = f'https://api.hubapi.com/crm/v3/properties/{object_type}'
#     headers = {
#         'authorization': "Bearer " + hubspot_access_token,
#     }
    
#     try:
#         async with session.get(url, headers=headers) as response:
#             if response.status == 200:
#                 data = await response.json()
#                 results = data.get('results', [])
#                 properties = []
#                 for prop in results:
#                     name = prop.get('name', '')
#                     print(name)
#                     description = prop.get('description', '')
#                     embedding_name = await get_embedding(name)
#                     embedding_description = await get_embedding(description)
#                     property_dict = {
#                         "name": name,
#                         "description": description,
#                         "embedding_name": embedding_name,
#                         "embedding_description": embedding_description,
#                         "label": prop.get('label'),
#                         "hidden": prop.get('hidden'),
#                         "archived": prop.get('archived')
#                     }
#                     properties.append(property_dict)
#                 print(properties)
#                 return {object_type: properties}
#             else:
#                 return {object_type: f"Failed to fetch properties, status code: {response.status}"}
#     except Exception as e:
#         print(f"Exception when calling HubSpot API: {str(e)}")
#         return {object_type: "Failed to fetch properties"}
    

# async def save_properties_to_chroma(properties):
#     collection = chroma_client.get_or_create_collection(name="properties")
#     for object_type, props in properties.items():
#         if not isinstance(props, list):
#             print(f"Skipping {object_type} due to error: {props}")
#             continue
#         for prop in props:
#             metadata = {
#                 "name": prop["name"],
#                 "description": prop["description"],
#                 "label": prop["label"],
#                 "hidden": prop["hidden"],
#                 "archived": prop["archived"]
#             }
#             embedding = prop["embedding_name"] + prop["embedding_description"]
#             await collection.add(embeddings=[embedding], metadatas=[metadata])


# async def save_properties():
#     async with aiohttp.ClientSession() as session:
#         types = ["contacts", "companies", "deals"]
#         tasks = [fetch_properties(session, object_type) for object_type in types]
#         properties = await asyncio.gather(*tasks)
#         await save_properties_to_chroma({k: v for d in properties for k, v in d.items()})


# asyncio.run(save_properties())
























# async def fetch_properties(session, object_type):
#     url = f'https://api.hubapi.com/crm/v3/properties/{object_type}'
#     headers = {
#         'authorization': "Bearer " + hubspot_access_token,
#     }
    
#     try:
#         async with session.get(url, headers=headers) as response:
#             if response.status == 200:
#                 data = await response.json()
#                 results = data.get('results', [])
#                 print(results)
#                 properties = [
#                     {
#                         "name": prop.get('name'),
#                         "label": prop.get('label'),
#                         "description": prop.get('description'),
#                         "hidden": prop.get('hidden'),
#                         "object_type": object_type,
#                     }
#                     for prop in results
#                 ]
#                 return properties
#             else:
#                 return {object_type: f"Failed to fetch properties, status code: {response.status}"}
#     except Exception as e:
#         print(f"Exception when calling HubSpot API: {str(e)}")
#         return {object_type: "Failed to fetch properties"}
    

# async def save_properties():
#     async with aiohttp.ClientSession() as session:
#         types = ["contacts", "companies", "deals"]
#         tasks = [fetch_properties(session, object_type) for object_type in types]
#         properties_lists = await asyncio.gather(*tasks)
        
#     properties = [prop for sublist in properties_lists for prop in sublist]
#     print(properties)
#     return properties




# async def embed_and_save_properties(properties):
#     collection = chroma_client.get_or_create_collection(name="properties")
#     print("Saving properties in collection: ", collection) 

#     async def embed_and_save(property):
#         unique_id = str(uuid.uuid4())
#         name = property["name"]
#         description = property["description"]

#         if name is None or description is None:
#             print(f"Skipping property {property['name']} due to missing name or description")
#             return

#         # Embedding name and description
#         embedding_name = await get_embedding(name)
#         embedding_description = await get_embedding(description)
        
#         # Preparing metadata and embedding
#         metadata = {
#                 "name": name if name is not None else "",
#                 "description": description if description is not None else "",
#                 "label": property["label"] if property["label"] is not None else "",
#                 "hidden": property["hidden"] if property["hidden"] is not None else False,
#                 "object_type": property["object_type"] if property["object_type"] is not None else ""
#             }
        
#         # Check for None values in the original property dictionary
#         none_keys = [key for key, value in property.items() if value is None]
#         if none_keys:
#             print(f"Property {property['name']} has None values in fields: {none_keys}")

#         embedding = embedding_name + embedding_description

#         # Saving to Chroma
#         loop = asyncio.get_running_loop()
#         await loop.run_in_executor(None, lambda: collection.add(embeddings=[embedding], metadatas=[metadata], ids=[unique_id]))

#     tasks = [embed_and_save(prop) for prop in properties]
#     await asyncio.gather(*tasks)


# async def main():
#     properties = await save_properties()
#     # await embed_and_save_properties(properties)

# asyncio.run(main())





















































# # fetch properties
# async def fetch_properties(object_type):
#     try:
#         if object_type == "contact":
#             properties = hubspot.crm.properties.core_api.get_all('contacts')
#         elif object_type == "company":
#             properties = hubspot.crm.properties.core_api.get_all('companies')
#         elif object_type == "deal":
#             properties = hubspot.crm.properties.core_api.get_all('deals')
#         else:
#             return {object_type: "Invalid object type"}
#         return properties.results if properties else []
#     except ApiException as e:
#         print("Exception when calling PropertiesApi->get: %s\n" % e)
#         return {object_type: "Failed to fetch properties"}


# # Embed the Properties
# async def save_properties():
#     properties = []
#     types = ["contact", "company", "deal"]
#     for type in types: 
#         property_list = []
#         property_instances = await fetch_properties(type)

#         for property_instance in property_instances:
#             property_dict = {
#                 "name": property_instance.name if hasattr(property_instance, 'name') else None,
#                 "label": property_instance.label if hasattr(property_instance, 'label') else None,
#                 "description": property_instance.description if hasattr(property_instance, 'description') else None,
#                 "hidden": property_instance.hidden if hasattr(property_instance, 'hidden') else None,
#                 "archived": property_instance.archived if hasattr(property_instance, 'archived') else None,
#             }
#             property_list.append(property_dict)
#         properties.append({type: property_list})
#     print(properties)
#     return properties





# async def main():
#     await save_properties()

# asyncio.run(main())






















































# old stuff

# chroma_client = chromadb.Client()
# collection = chroma_client.create_collection(name="TranscriptChunks")


# def save_to_chroma(embeddings, chunked_transcript, collection_name="my_collection"):
#     print("save to chroma")
#     # Check if collection exists, if not create one
#     if collection_name not in [coll.name for coll in chroma_client.list_collections()]:
#         collection = chroma_client.create_collection(name=collection_name)
#     else:
#         collection = chroma_client.get_collection(name=collection_name)
    
#     # generate ids for each chunk
#     ids = [generate_id(chunk, idx) for idx, chunk in enumerate(chunked_transcript)]

#     # create metadatas with order for each chunk
#     metadata = [{"source": "transcript_chunk", "order": idx} for idx, _ in enumerate(chunked_transcript)]
    
#     # Save the embeddings and documents to ChromaDb
#     collection.add(
#         embeddings=embeddings,
#         documents=chunked_transcript,
#         metadatas=metadata,
#         ids=ids
#     )
#     print(f"Saved {len(chunked_transcript)} embeddings to ChromaDb in collection '{collection_name}'")














    # # Initialize Pinecone index
    # pinecone.init(      
    #     api_key='7179c33c-f1d5-4d47-b15b-c17c1cb69b97',      
    #     environment='gcp-starter'      
    # )    
    # index = pinecone.Index('chatpdf')



        # pinecone.init(      
    #     api_key='7179c33c-f1d5-4d47-b15b-c17c1cb69b97',      
    #     environment='gcp-starter'      
    # )    




        #prepeare for the Pinecone upsert
    # records = [(id_, embedding, {"transcript_chunk": chunk}) for id_, embedding, chunk in zip(ids, embeddings, chunked_transcript)]