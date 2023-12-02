from fastapi import FastAPI, APIRouter, status, Depends, HTTPException
import logging
import json
import os
import openai
from slack_sdk import WebClient
import threading
from app.custom_functions import create_ssh_user, get_k8s_namespaces, get_secrets, get_k8s_pods, get_k8s_events
from ..config import settings
import sentry_sdk
from .. import schemas
from sqlalchemy.orm import Session
from typing import List

from sqlalchemy import func
# from sqlalchemy.sql.functions import func
from .. import models, schemas, oauth2
from ..database import get_db

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Slack']
)

sentry_sdk.init(
    dsn=os.getenv(settings.sentry_dsn),
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

slack_client = WebClient(token=settings.slack_token)
app = FastAPI()


def send_message_to_chatgpt(message):
    """
    - sends conversation buffer with message from user to openai and gets back the response 
    - if response have a function call, call that function and register the result in conversation buffer and send the conversation buffer back to openai. 
    - finally when no function call in the result, return the response.
    """

    global conversation_buffer
    openai.api_key = settings.openai_api_key
    response = ""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            functions=[
                {
                    "name": "create_ssh_user",
                    "description": "create ssh user in server",
                    "parameters": {
                            "type": "object",
                            "properties": {
                                "username": {"type": "string", "description": "username of user"},
                                "serverIP": {"type": "string", "description": "serverIP of server"},
                                "sshPubKey": {"type": "string", "description": "ssh public key of user"},
                            },
                        "required": ["username", "serverIP", "sshPubKey"],
                    },
                },
                {
                    "name": "get_k8s_namespaces",
                    "description": "get all namespaces name in a list format for the current cluster context",
                    "parameters": {"type": "object", "properties": {}, },
                },
                {
                    "name": "get_k8s_events",
                    "description": "get the events in a table format for the current cluster context",
                    "parameters": {"type": "object", "properties": {}, },
                },
                {
                    "name": "get_k8s_pods",
                    "description": "get the pods in a list format for the current cluster context",
                    "parameters": {"type": "object", "properties": {}, },
                },
                {
                    "name": "get_secrets",
                    "description": "get the secrets in a table format",
                    "parameters": {"type": "object", "properties": {}, },
                }
            ]
        )
    except Exception as exc:
        logger.error(f"bad openai response: {exc}")

    if response.choices[0].message:

        message = response.choices[0].message
        if 'function_call' in message:
            logger.info(
                f"a function call has been detected for {message['function_call']['name']}")
            available_functions = {
                "create_ssh_user": create_ssh_user,
                "get_k8s_namespaces": get_k8s_namespaces,
                "get_k8s_events": get_k8s_events,
                "get_k8s_pods": get_k8s_pods,
                "get_secrets": get_secrets
            }
            function_name = message['function_call']['name']
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(message["function_call"]["arguments"])
            if function_name == "create_ssh_user":
                function_response = fuction_to_call(
                    username=function_args.get("username"),
                    serverIP=function_args.get("serverIP"),
                    sshPubKey=function_args.get("sshPubKey")
                )
            if function_name == "get_k8s_namespaces" or function_name == "get_k8s_events" or function_name == "get_k8s_pods" or function_name == "get_secrets":
                function_response = fuction_to_call()

            conversation_buffer.append(
                {"role": "function", "name": function_name, "content": function_response})
            gpt_function_response = send_message_to_chatgpt(
                conversation_buffer)
            return gpt_function_response
    logger.debug(f"conversation buffer with openai: {conversation_buffer}")
    return response


def send_response_to_slack(response, channel_id, event_ts):
    """send response to slack returned from openai"""
    response = slack_client.chat_postMessage(
        channel=channel_id,
        text=response,
        thread_ts=event_ts
    )
    logger.debug(f"slack reponse {response}")
    return response


openai.api_key = settings.openai_api_key
conversation_buffer = []
conversation_buffer.append(
    {"role": "system", "content": "you are a devops engineer"})


def handle_chat(event):
    """handle chat event from slack, interact with openai and send final response to slack"""
    global conversation_buffer
    if event.event.type and event.event.type != 'test':
        if "goodbye" in event.event.text:
            logger.info(
                "clearing conversation buffer because 'goodbye' received")
            conversation_buffer = []
            slack_response = send_response_to_slack(
                "Have a good day!", event.event.channel, event.event.event_ts)
            return {"response": "Have a good day!"}
        else:
            user_message = event.event.text
            content = "user: " + user_message
            conversation_buffer.append({"role": "user",   "content": content})
            logger.info(
                f"sending conversation buffer to chatgpt: {conversation_buffer}")
            gpt_response = send_message_to_chatgpt(
                conversation_buffer).choices[0].message
            if gpt_response.content:
                content = gpt_response.content
                conversation_buffer.append(
                    {"role": "assistant", "content": content})
                logger.info(
                    f"got a response from chatgpt, sending response to slack: {content}")
                send_response_to_slack(
                    content, event.event.channel, event.event.event_ts)


@router.post("/slack_events", status_code=status.HTTP_200_OK)
async def receive_slack_event(event: schemas.SlackPayload, db: Session = Depends(get_db)):

    logger.info(f"event received from slack: {event}")
    if event.token != settings.slack_event_token:
        raise HTTPException(status_code=403, detail="Invalid token")
    if event.type == "url_verification":
        return {"challenge": event.challenge}
    # new_event = models.SlackEvent(**event.event.dict())
    # db.add(new_event)
    # db.commit()
    # db.refresh(new_event)
    if event.event.user != settings.slack_bot_user:
        threading.Thread(target=handle_chat, args=(event,)).start()
        return {"message": "Event received"}
