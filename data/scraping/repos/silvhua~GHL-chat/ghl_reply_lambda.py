import json
import sys
from chat_functions import *
from ghl_requests import *
from langchain.agents import Tool
def lambda_handler(event, context):
    """
    This Lambda function is triggered by another function when the payload type is 'InboundMessage'.
    """

    response = refresh_token()
    if response['statusCode'] // 100 == 2:
        # Extract the payload from the event
        payload = event
        print(f'Payload: {payload}')
        contactId = payload.get('contactId')
        # contactId = "C3a796qLEF4HtDjvKJ1K"
        InboundMessage = payload.get('body')
        locationId = payload.get('locationId', 'SAMLab')
        location = os.getenv(locationId)
        print(f'location: {location}')

        system_message_dict = dict()
        conversation_dict = dict()
        conversation_id = 1

        tools = [
            Tool(
                name=f"foo-{i}",
                func=fake_func,
                description=f"a silly function that you can use to get more information about the number {i}",
            )
            for i in range(2)
        ]
        business_dict = {
            'SAM_Lab': (
                'SAM_Lab.txt', # business-specific system message
                'SAM_Lab_examples.txt', # Examples 
                'SAM_Lab_doc.md' # RAG doc
                )
        }

        try:
            refresh_token_response = refresh_token()
            # print(f'Refresh token response: {refresh_token_response}')

            system_message_dict[conversation_id] = create_system_message(
                'SAM_Lab', business_dict, prompts_filepath='prompts',
                examples_filepath='chat_examples', doc_filepath='rag_docs'
            )
            conversation_dict[conversation_id] = create_chatbot(
                contactId, system_message_dict[conversation_id], tools=tools,
                # model='gpt-4-32k'
                )

            reply = chat_with_chatbot(
                InboundMessage, conversation_dict[conversation_id]
            )
            print(f'Reply from `chat_with_chatbot`: {reply}')
            ghl_api_response = ghl_request(
                contactId=contactId, 
                text=reply['output'],
                endpoint='createTask', 
                payload=None, 
                location=location
            )
            if ghl_api_response['status_code'] // 100 == 2:
                message = f'Created task for contactId {contactId}: \n{ghl_api_response}\n'
            else:
                message = f'Failed to create task for contactId {contactId}: \n{ghl_api_response}\n'
                message += f'Status code: {ghl_api_response["status_code"]}. \nResponse reason: {ghl_api_response["response_reason"]}'
            print(message)
            return {
                'statusCode': 200,
                'body': json.dumps(message)
            }
        except Exception as error:
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            message = f"Error in line {lineno} of {filename}: {str(error)}"
            print(message)
            return {
                'statusCode': 500,
                'body': json.dumps(message)
            }

    else:
        print('Failed to refresh token')
        return response