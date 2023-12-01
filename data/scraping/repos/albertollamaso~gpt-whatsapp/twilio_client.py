import os
from twilio.rest import Client
from openai_client import openai_get_response


def twilio_send_message(source_msg, source_phone):

    account_sid = os.getenv('TW_ACCOUNT_SID')
    auth_token = os.getenv('TW_AUTH_TOKEN')
    client = Client(account_sid, auth_token)

    body = openai_get_response(source_msg)

    message = client.messages.create(
                                body=body,
                                from_=os.getenv('TW_PHONE_NUMBER'),
                                to=source_phone
                            )

    print("response: " + message.body)
