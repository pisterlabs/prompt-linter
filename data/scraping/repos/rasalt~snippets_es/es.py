import json

from typing import List
from langchain.schema import Document

from google.cloud import discoveryengine_v1beta
from google.protobuf.json_format import MessageToDict

default_max_extractive_answer_count = 5
default_max_extractive_segment_count = 1
default_query_expansion_condition = 1


def es_raw_search_result(result):
    for i in result:
        print("-"*79)
        print(i)
        print("-"*79)

from google.cloud import discoveryengine_v1beta
def converse_conversation(formatted_parent_conversation, userquery):
    
    print("Formatted parent conversation string: {}".format(formatted_parent_conversation))
    conversational_search_service_client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    if 1:
        query = discoveryengine_v1beta.TextInput(input = userquery)
        request = discoveryengine_v1beta.ConverseConversationRequest(
                name = formatted_parent_conversation,
                query = query
        )
    else:        
        request = {}
        request['query'] = query
        request['name'] = formatted_parent_conversation

     # Call the API and handle any network failures.
    try:
        response = conversational_search_service_client.converse_conversation(request)
        print(f'Response data: {response}')
    except Exception as ex:
        print(f'Call failed with message: {str(ex)}')

    return response

def create_conversation(PROJECT, LOCATION, DATA_STORE):
    from google.cloud import discoveryengine_v1beta
    # Get the list of methods
    if 0:
        methods = dir(discoveryengine_v1beta.ConversationalSearchServiceClient)
    # Print the methods
        for method in methods:
            print(method)
  
    formatted_parent = discoveryengine_v1beta.ConversationalSearchServiceClient.data_store_path(
       PROJECT,
       LOCATION,
       DATA_STORE
    )
    # Create a client.
    conversational_search_service_client =    discoveryengine_v1beta.ConversationalSearchServiceClient()

    # Prepare the request message.
    conversation = discoveryengine_v1beta.Conversation()
    if 1:
        request = discoveryengine_v1beta.CreateConversationRequest(
            parent = formatted_parent,
            conversation = conversation
        )
    # Call the API and handle any network failures.
    try:
        response = conversational_search_service_client.create_conversation(request)
        print(f'Response data: {response}')
    except Exception as ex:
        print(f'Call failed with message: {str(ex)}')

    formatted_parent_conversation = response.name
    print("Formatted convo name is: {}".format(formatted_parent_conversation))
    return formatted_parent_conversation

def es_raw_search_summary(project,
              search_engine,
              query,
              filtername = None,           
              location="global",
              serving_config_id='default_config',                          
              ):
    
    search_client = discoveryengine_v1beta.SearchServiceClient()
    serving_config: str = search_client.serving_config_path(
        project=project,
        location=location,
        data_store=search_engine,
        serving_config=serving_config_id
    )
    content_search_spec = {
        "extractive_content_spec": {
            "max_extractive_answer_count": default_max_extractive_answer_count,
        },
        "extractive_content_spec": {
            "max_extractive_segment_count": default_max_extractive_segment_count,
        }
    }
    content_search_spec = {
        "summary_spec": {
            "summary_result_count": 5
        },
        "extractive_content_spec": {
            "max_extractive_answer_count" : 1,
            "max_extractive_segment_count": default_max_extractive_segment_count,
        },
        "snippet_spec":
        {
            "max_snippet_count": 1
        }
    }    
    query_expansion_spec = {
            "condition": default_query_expansion_condition
    }
    #print("Filter is :{}".format(filtername))
    request = discoveryengine_v1beta.SearchRequest(
            query=query,
            filter=filtername,
            serving_config=serving_config,
            page_size=5,
            content_search_spec=content_search_spec,
            query_expansion_spec=query_expansion_spec,
        )

    result = search_client.search(request)
    return result.summary.summary_text, result