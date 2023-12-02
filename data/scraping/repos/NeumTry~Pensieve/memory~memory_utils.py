from typing import List, Union
from abc import ABC
import streamlit as st

def create_user(user:str):
    import weaviate
    url = st.secrets["Weaviate_URL"]
    api_key = st.secrets["Weaviate_Key"]
    class_name = "User_" + user.replace("-","_")
    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    )
    try:
        client.schema.create_class({
            "class": class_name
        })
    except weaviate.UnexpectedStatusCodeException as e:
        if 'already exists' not in e.message: # We got an error that is nto because the class already exists. can we check the class before maybe?. In this case, we should throw
            print(f"Error when creating class in weaviate: {e}")
            raise e

def store_messages(messages:List[dict], user:str, session:str, message_id:str):
    import weaviate
    from langchain.embeddings import OpenAIEmbeddings
    url = st.secrets["Weaviate_URL"]
    api_key = st.secrets["Weaviate_Key"]
    class_name = "User_" + user.replace("-","_")
    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    )

    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_Key"])
     
    with client.batch.configure(
        batch_size=50,
        num_workers=2,
    ) as batch:
        for i in range(0,len(messages)):
            properties = {
                **messages[i], 
                **{"session":session}
            }
            embedding = embeddings_model.embed_documents(texts=[messages[i]['content']])
            batch.add_data_object(
                data_object=properties,
                class_name=class_name,
                vector=embedding[0],
                uuid=message_id
            )
    return len(messages)

def search_chat_memory(query:str, user:str, session:str) -> List:
    import weaviate
    from langchain.embeddings import OpenAIEmbeddings
    url = st.secrets["Weaviate_URL"]
    api_key = st.secrets["Weaviate_Key"]
    class_name = "User_" + user.replace("-","_")
    
    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    )

    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_Key"])
    embedding = embeddings_model.embed_query(text=query)

    matches = []
    try:
        search_result = (
            client.query
            .get(class_name=class_name, properties=['content', 'role'])
            .with_near_vector(content = {
                'vector' : embedding,
                'certainty' : 0.7
            })
            .with_where({
                "operator": "And",
                "operands": [
                    {
                        "path":["session"],
                        "operator": "Equal",
                        "valueText":session
                    },
                    {
                        "path":["role"],
                        "operator": "NotEqual",
                        "valueText":"summary"
                    }
                ]
            })
            .with_limit(5)
            .with_additional(['certainty'])
            .do()
        )
        for result in search_result["data"]["Get"][class_name]:
            # unify our api with the metadata.. or just return whatever metadata we have. (?)
            matches.append(result)
    except Exception as e:
        print(f"No chat history retrieved")
        return []
    
    return matches

def search_document_memory(query:str, user:str, session:str) -> List:
    import weaviate
    from langchain.embeddings import OpenAIEmbeddings
    url = st.secrets["Weaviate_URL"]
    api_key = st.secrets["Weaviate_Key"]
    
    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    )

    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_Key"])
    embedding = embeddings_model.embed_query(text=query)

    matches = []
    try:
        class_name_2 = "User_" + user.replace("-","_") + "_documents"
        search_result_2 = (
            client.query
            .get(class_name=class_name_2, properties=['text'])
            .with_near_vector(content = {
                'vector' : embedding,
                'certainty' : 0.7
            })
            .with_limit(5)
            .with_additional(['certainty'])
            .do()
        )
        for result in search_result_2["data"]["Get"][class_name_2]:
            # unify our api with the metadata.. or just return whatever metadata we have. (?)
            matches.append(result)
    except Exception as e:
        print(f"No document history retrieved")
        return []
    
    return matches

def search_full_memory_and_rerank(query:str, user:str, session:str):
    chat_history_results = search_chat_memory(query=query, user=user, session=session)
    document_results = search_document_memory(query=query, user=user, session=session)
    all_results = chat_history_results + document_results

    # Normalize results to just text and certainty
    normalized_all_results = []
    for result in all_results:
        if 'content' in result.keys():
            normalized_all_results.append({
                "text":result['content'],
                "score":result['_additional']['certainty']
            })
        elif 'text' in result.keys():
            normalized_all_results.append({
                "text":result['text'],
                "score":result['_additional']['certainty']
            })    

    # Re-rank based on certainty and return back only top 5 context pieces total
    sorted_results = sorted(normalized_all_results, key= lambda x: x['score'], reverse=True)
    top_5_results = sorted_results[:5]
    return top_5_results

    # More re ranking can be added like weighting chat vs document results differently

def generate_and_store_summary(messages:List, user:str, session:str):
    import openai
    import uuid
    system_message = {"role": "system", "content": "Summarize the conversation into 1-2 paragraphs."}
    summary = openai.ChatCompletion.create(
        model="gpt-4",
        messages= [system_message] + messages
    ).choices[0].message.content
    message = {
        "role":"summary",
        "content":summary
    }
    store_messages(messages=[message],user=user, session=session, message_id=uuid.uuid4())

def get_summary_and_generate_message(user:str, session:str):
    import weaviate
    from langchain.embeddings import OpenAIEmbeddings
    url = st.secrets["Weaviate_URL"]
    api_key = st.secrets["Weaviate_Key"]
    class_name = "User_" + user.replace("-","_")
    
    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    )

    matches = []
    try:
        search_result = (
            client.query
            .get(class_name=class_name, properties=['content', 'role'])
            .with_where({
                "operator": "And",
                "operands": [
                    {
                        "path":["session"],
                        "operator": "Equal",
                        "valueText":session
                    },
                    {
                        "path":["role"],
                        "operator": "Equal",
                        "valueText":"summary"
                    }
                ]
            })
            .with_limit(1)
            .do()
        )
        if(len(search_result["data"]["Get"][class_name]) == 0 ):
            raise Exception("No summary")
        for result in search_result["data"]["Get"][class_name]:
            # unify our api with the metadata.. or just return whatever metadata we have. (?)
            matches.append(result)
    except Exception as e:
        print(f"No summary found")
        return None
    
    return matches[0]['content']