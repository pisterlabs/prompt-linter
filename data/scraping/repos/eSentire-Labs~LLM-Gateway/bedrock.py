
"""
This is the primary FastAPI router where LLM endpoints are proxied and monitored.

This is a barebones bedrock POC and further development is required to use.

"""
# pylint: disable=too-many-locals,too-many-arguments,dangerous-default-value
import datetime
import json
import os
import uuid
import requests
from typing import Optional

# External Dependencies
import boto3
import botocore
from botocore.config import Config
from fastapi import Depends, APIRouter
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from src.constants import  AWS_REGION, BEDROCK, BEDROCK_ASSUMED_ROLE, CONTENT_TYPE, ENABLE_SUBM_API, LLM_TYPE, MODEL_ID, URL
from src.modules.common import RequestException, log_to_esentire
from src.modules.database import get_db, uuid_exists
from src.modules.logger import logger
from src.modules.auto_models import ChatgptLog, ImageLog, MetaSummarizerLog
from src.modules.classes import (
    ChatBedrockJson,
    unknown_error,
    chat_success,
    scope_error,
    dns_error,
)

#
# Source: https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/utils/bedrock.py
#
def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


bedrock_router = APIRouter()
# Only display this endpoint if sagemaker is enabled
if BEDROCK in LLM_TYPE:

    bedrock_runtime = get_bedrock_client(
        assumed_role=BEDROCK_ASSUMED_ROLE,
        region=AWS_REGION
    )

    ai21_llm = Bedrock(model_id=MODEL_ID, client=bedrock_runtime)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=ai21_llm, verbose=True, memory=memory
    )


    # LLM endpoints proxied to in this API
    @bedrock_router.post(
        "/chat_br",
        tags=["Bedrock"],
        responses={200: chat_success, 400: unknown_error, 500: dns_error, 403: scope_error},
    )
    async def submit_chat_br(
        json_object: ChatBedrockJson,
        db_engine: Session = Depends(get_db),
    ):
        """
        This is an endpoint to interact with a bedrock llm model.
        """
        # pylint: disable=protected-access
        json_object = json_object.dict(exclude_unset=True)
        
        modelId = json_object.pop("modelId")

        logger.info("json_object: %s", json_object)

        # This generates a uuid to track each record in the db
        while True:
            uuid_generated = uuid.uuid4()
            if uuid_exists(uuid_generated, db_engine):
                continue
            break

        # Extract the convo_title if passed alongside request, so we don't pass downstream to llm, since this is an internal field
        convo_title=""
        if "convo_title" in json_object:
            convo_title = json_object.pop("convo_title")
        else:
            convo_title = json_object["prompt"][:50]

        # Extract the root_id if passed alongside request, so we don't pass downstream to llm, since this is an internal field
        gpt_root_id= None
        if "root_id" in json_object:
            gpt_root_id = json_object.pop("root_id")

        logger.info("uuid: %s", uuid_generated)
        logger.info("convo_title: %s", convo_title)
                
        try:
            time_submitted = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            body = json.dumps(json_object)
            logger.info("body: %s", body)
            response = bedrock_runtime.invoke_model(
                body=body, modelId=modelId, accept=CONTENT_TYPE, contentType=CONTENT_TYPE
            )
            
            logger.info("response: %s", response)
            response_body = json.loads(response.get("body").read())

            prompt_response = response_body.get("completions")[0].get("data").get("text")

        except botocore.exceptions.ClientError as error:

            if error.response['Error']['Code'] == 'AccessDeniedException':
                print(f"\x1b[41m{error.response['Error']['Message']}\
                        \nTo troubeshoot this issue please refer to the following resources.\
                        \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                        \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")

            else:
                raise error


        logger.info("Configuration of request: %s", json_object)
        logger.info("Response: %s", prompt_response)

        chat_log = ChatgptLog(
            id=uuid_generated,
            request=json.dumps(json_object), 
            response=json.dumps(response_body),
            usage_info=json.dumps(response_body["prompt"]["tokens"]),
            user_name = "user", # Replace with the actual username of whomever sent the request using whichever IAM tools you prefer
            title = "tile", # Replace with user job title-- this is an example of additional metadata that might be useful to capture, or to use in your dashboards to create interesting job role based reports
            convo_title=convo_title, # This is used in the chat history feature so users can quickly get an idea of prior conversation content
            root_gpt_id= "1" # This is used to draw a lineage between different interactions so we can trace a single conversation
        )

        # Store this primary
        db_engine.add(chat_log)
        db_engine.commit()

        if ENABLE_SUBM_API=="true":
            try:
                print("logging to esentire")
                esentire_response = log_to_esentire(
                    raw_request=json_object["prompt"],
                    raw_response=prompt_response,
                    associated_users= ["example_user_id"],
                    time_submitted= time_submitted,
                    associated_devices= ["device1, device2"],
                    associated_software= ["software1","software2"]
                    )
            except Exception as error:
                print("Error logging to esentire in submit_chat: {error}")
                
        # # Return the generated text as a response to the client
        return response_body

    @bedrock_router.post(
        "/chat_br_langchain",
        tags=["Bedrock"],
        responses={200: chat_success, 400: unknown_error, 500: dns_error, 403: scope_error},
    )
    async def submit_chat_br_langchain(
        json_object: ChatBedrockJson,
        db_engine: Session = Depends(get_db),
    ):
        """
        This is the primary endpoint specifically to retain conversation context in a chatbox using bedrock.
        Excerpts of the code came from the Notebook shared here: https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/04_Chatbot/00_Chatbot_AI21.ipynb
        """
        # pylint: disable=protected-access
        json_object = json_object.dict(exclude_unset=True)
        
        modelId = json_object.pop("modelId")

        logger.info("json_object: %s", json_object)

        # This generates a uuid to track each record in the db
        while True:
            uuid_generated = uuid.uuid4()
            if uuid_exists(uuid_generated, db_engine):
                continue
            break

        # Extract the convo_title if passed alongside request, so we don't pass downstream to llm, since this is an internal field
        convo_title=""
        if "convo_title" in json_object:
            convo_title = json_object.pop("convo_title")
        else:
            convo_title = json_object["prompt"][:50]

        # Extract the root_id if passed alongside request, so we don't pass downstream to llm, since this is an internal field
        gpt_root_id= None
        if "root_id" in json_object:
            gpt_root_id = json_object.pop("root_id")

        logger.info("uuid: %s", uuid_generated)
        logger.info("convo_title: %s", convo_title)
        
        
        try:
            time_submitted = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            prompt = json_object["prompt"]
            logger.info("body: %s", prompt)
            
            response = conversation.predict(input=prompt)
            logger.info("response: %s", response)

        except ValueError as error:
            if  "AccessDeniedException" in str(error):
                print(f"\x1b[41m{error}\
                \nTo troubeshoot this issue please refer to the following resources.\
                \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
                class StopExecution(ValueError):
                    def _render_traceback_(self):
                        pass
                raise StopExecution        
            else:
                raise error


        logger.info("Configuration of request: %s", json_object)
        logger.info("Response: %s", response)# prompt_response)

        chat_log = ChatgptLog(
            id=uuid_generated,
            request=json.dumps(json_object), 
            response=response, 
            # no usage_info
            # usage_info=json.dumps(response_body["prompt"]["tokens"]),
            user_name = "user", # Replace with the actual username of whomever sent the request using whichever IAM tools you prefer
            title = "tile", # Replace with user job title-- this is an example of additional metadata that might be useful to capture, or to use in your dashboards to create interesting job role based reports
            convo_title=convo_title, # This is used in the chat history feature so users can quickly get an idea of prior conversation content
            root_gpt_id= "1" # This is used to draw a lineage between different interactions so we can trace a single conversation
        )

        # Store this primary
        db_engine.add(chat_log)
        db_engine.commit()

        if ENABLE_SUBM_API=="true":
            try:
                print("logging to esentire")
                esentire_response = log_to_esentire(
                    raw_request=json_object["prompt"],
                    raw_response=prompt_response,
                    associated_users= ["example_user_id"],
                    time_submitted= time_submitted,
                    associated_devices= ["device1, device2"],
                    associated_software= ["software1","software2"]
                    )
            except Exception as error:
                print("Error logging to esentire in submit_chat: {error}")
                
        # # Return the generated text as a response to the client
        return response
